import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
LOCAL_CLIP_PARENT = os.path.join(
    REPO_ROOT, "maskrcnn_benchmark", "modeling", "roi_heads", "relation_head"
)

if LOCAL_CLIP_PARENT not in sys.path:
    sys.path.insert(0, LOCAL_CLIP_PARENT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute HOLa-style low-rank factors from predicate descriptions "
            "encoded by the local CLIP text encoder."
        )
    )
    parser.add_argument(
        "--predicate-json",
        type=str,
        default=None,
        help="Path to dataset dict JSON containing idx_to_predicate or predicate_to_idx.",
    )
    parser.add_argument(
        "--descriptions",
        type=str,
        default=None,
        help="Optional JSON/CSV file with predicate descriptions.",
    )
    parser.add_argument(
        "--predicate-list",
        type=str,
        default=None,
        help='Optional comma-separated predicate list for smoke tests, e.g. "on,under,riding".',
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-B/32",
        help="Local CLIP checkpoint path or built-in model name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for text encoding.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="Target low-rank dimension m.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for CLIP text encoding.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="L2-normalize CLIP text features before decomposition. Enabled by default.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize",
        help="Disable L2-normalization before decomposition.",
    )
    parser.add_argument(
        "--keep-background",
        action="store_true",
        help="Include __background__ in the decomposition.",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="a photo of {predicate}",
        help="Fallback prompt when a predicate description is missing.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(REPO_ROOT, "outputs", "hola_predicate_factors.pt"),
        help="Output .pt path for the computed factors.",
    )
    return parser.parse_args()


def load_predicates_from_json(path: str) -> List[Tuple[int, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if "idx_to_predicate" in payload:
        idx_to_predicate = payload["idx_to_predicate"]
        if isinstance(idx_to_predicate, dict):
            items = sorted(
                ((int(index), name) for index, name in idx_to_predicate.items()),
                key=lambda item: item[0],
            )
        else:
            items = [(index, name) for index, name in enumerate(idx_to_predicate)]
        return items

    if "predicate_to_idx" in payload:
        predicate_to_idx = payload["predicate_to_idx"]
        items = sorted(
            ((int(index), name) for name, index in predicate_to_idx.items()),
            key=lambda item: item[0],
        )
        return items

    raise ValueError(
        f"No predicate mapping found in {path}. Expected `idx_to_predicate` or `predicate_to_idx`."
    )


def load_predicates_from_list(raw_predicates: str) -> List[Tuple[int, str]]:
    predicates = [item.strip() for item in raw_predicates.split(",") if item.strip()]
    if not predicates:
        raise ValueError("`--predicate-list` is empty after parsing.")
    return list(enumerate(predicates))


def normalize_key(text: str) -> str:
    return " ".join(text.strip().lower().split())


def load_description_map(path: str) -> Dict[str, str]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Description JSON must be an object mapping predicate -> description.")
        return {normalize_key(key): str(value).strip() for key, value in payload.items()}

    if ext == ".csv":
        description_map: Dict[str, str] = {}
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required_fields = {"predicate", "description"}
            if not reader.fieldnames or not required_fields.issubset(reader.fieldnames):
                raise ValueError("Description CSV must contain `predicate` and `description` columns.")
            for row in reader:
                predicate = normalize_key(row["predicate"])
                description = row["description"].strip()
                if predicate:
                    description_map[predicate] = description
        return description_map

    raise ValueError(f"Unsupported description format: {path}. Use JSON or CSV.")


def resolve_descriptions(
    predicates: Sequence[Tuple[int, str]],
    description_map: Optional[Dict[str, str]],
    prompt_template: str,
) -> Tuple[List[str], List[bool]]:
    descriptions: List[str] = []
    fallback_mask: List[bool] = []

    for _, predicate_name in predicates:
        lookup_key = normalize_key(predicate_name)
        description = ""
        if description_map is not None:
            description = description_map.get(lookup_key, "").strip()

        used_fallback = not description
        if used_fallback:
            description = prompt_template.format(predicate=predicate_name)

        descriptions.append(description)
        fallback_mask.append(used_fallback)

    return descriptions, fallback_mask


def encode_text_features(
    texts: Sequence[str],
    clip_model_name: str,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    try:
        from CLIP import clip  # noqa: E402
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Failed to import the local CLIP package dependencies. "
            "Please install the missing package in your runtime environment "
            f"(missing: {error.name})."
        ) from error

    try:
        model, _ = clip.load(clip_model_name, device=device)
    except RuntimeError as error:
        raise RuntimeError(
            "Failed to load CLIP model. If the weights are not cached locally, "
            "pass `--clip-model /path/to/ViT-B-32.pt` or prepare the CLIP cache first. "
            f"Original error: {error}"
        ) from error

    model.eval()
    features: List[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start:start + batch_size])
            tokens = clip.tokenize(batch_texts).to(device)
            batch_features = model.encode_text(tokens).float().cpu()
            features.append(batch_features)

    return torch.cat(features, dim=0)


def truncate_rank(rank: int, num_predicates: int, feature_dim: int) -> int:
    max_rank = min(num_predicates, feature_dim)
    if max_rank < 1:
        raise ValueError("Need at least one predicate and one feature dimension for decomposition.")
    return min(rank, max_rank)


def compute_low_rank_factors(features: torch.Tensor, rank: int) -> Dict[str, torch.Tensor]:
    u, singular_values, vh = torch.linalg.svd(features, full_matrices=False)
    effective_rank = truncate_rank(rank, features.shape[0], features.shape[1])

    truncated_u = u[:, :effective_rank]
    truncated_s = singular_values[:effective_rank]
    truncated_vh = vh[:effective_rank, :]

    w = truncated_u * truncated_s.unsqueeze(0)
    m = truncated_vh
    reconstructed = w @ m

    squared = singular_values.pow(2)
    explained_variance_ratio = squared / squared.sum().clamp_min(1e-12)
    reconstruction_error = torch.norm(features - reconstructed, p="fro")

    return {
        "W": w,
        "M": m,
        "reconstructed": reconstructed,
        "rank": torch.tensor(effective_rank, dtype=torch.int64),
        "singular_values": singular_values,
        "explained_variance_ratio": explained_variance_ratio,
        "reconstruction_error": reconstruction_error,
    }


def filter_background(
    predicates: Sequence[Tuple[int, str]],
    keep_background: bool,
) -> List[Tuple[int, str]]:
    if keep_background:
        return list(predicates)
    return [
        (index, name)
        for index, name in predicates
        if normalize_key(name) != "__background__"
    ]


def ensure_parent_dir(path: str) -> None:
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def metadata_path(output_path: str) -> str:
    root, _ = os.path.splitext(output_path)
    return root + ".json"


def main() -> None:
    args = parse_args()

    if args.rank < 1:
        raise ValueError("--rank must be a positive integer.")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be a positive integer.")

    if not args.predicate_json and not args.predicate_list:
        raise ValueError("Provide either `--predicate-json` or `--predicate-list`.")

    if args.predicate_json and args.predicate_list:
        raise ValueError("Use only one of `--predicate-json` or `--predicate-list`.")

    if args.predicate_json:
        predicates = load_predicates_from_json(args.predicate_json)
    else:
        predicates = load_predicates_from_list(args.predicate_list)

    predicates = filter_background(predicates, keep_background=args.keep_background)
    if not predicates:
        raise ValueError("No predicates remain after background filtering.")

    description_map = None
    if args.descriptions:
        description_map = load_description_map(args.descriptions)

    descriptions, fallback_mask = resolve_descriptions(
        predicates=predicates,
        description_map=description_map,
        prompt_template=args.prompt_template,
    )

    features = encode_text_features(
        texts=descriptions,
        clip_model_name=args.clip_model,
        device=args.device,
        batch_size=args.batch_size,
    )

    if args.normalize:
        features = F.normalize(features, dim=-1)

    factor_payload = compute_low_rank_factors(features, args.rank)

    predicate_indices = [index for index, _ in predicates]
    predicate_names = [name for _, name in predicates]

    output_payload = {
        "features": features,
        "W": factor_payload["W"],
        "M": factor_payload["M"],
        "reconstructed": factor_payload["reconstructed"],
        "predicate_names": predicate_names,
        "predicate_indices": predicate_indices,
        "descriptions": descriptions,
        "rank": int(factor_payload["rank"].item()),
        "singular_values": factor_payload["singular_values"],
        "explained_variance_ratio": factor_payload["explained_variance_ratio"],
        "reconstruction_error": float(factor_payload["reconstruction_error"].item()),
    }

    metadata = {
        "predicate_json": args.predicate_json,
        "descriptions": args.descriptions,
        "clip_model": args.clip_model,
        "device": args.device,
        "normalize": args.normalize,
        "requested_rank": args.rank,
        "effective_rank": int(factor_payload["rank"].item()),
        "num_predicates": len(predicate_names),
        "feature_dim": int(features.shape[1]),
        "fallback_count": sum(fallback_mask),
        "fallback_mask": fallback_mask,
        "predicate_names": predicate_names,
        "predicate_indices": predicate_indices,
        "description_texts": descriptions,
        "reconstruction_error": float(factor_payload["reconstruction_error"].item()),
        "singular_values": factor_payload["singular_values"].tolist(),
        "explained_variance_ratio": factor_payload["explained_variance_ratio"].tolist(),
    }

    ensure_parent_dir(args.output)
    torch.save(output_payload, args.output)

    meta_output_path = metadata_path(args.output)
    ensure_parent_dir(meta_output_path)
    with open(meta_output_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"Saved HOLa predicate factors to: {args.output}")
    print(f"Saved metadata to: {meta_output_path}")
    print(f"Predicates: {len(predicate_names)}")
    print(f"Feature dim: {features.shape[1]}")
    print(f"Requested rank: {args.rank}")
    print(f"Effective rank: {int(factor_payload['rank'].item())}")
    print(f"Fallback descriptions: {sum(fallback_mask)}")
    print(f"Reconstruction error: {float(factor_payload['reconstruction_error'].item()):.6f}")


if __name__ == "__main__":
    main()
