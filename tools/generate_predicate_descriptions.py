#!/usr/bin/env python3
import argparse
import ast
import json
import os
import time
from collections import OrderedDict


PROMPT_TEMPLATE = """You are helping build an open-vocabulary scene graph generation model.

I will provide:
1. A visual relationship predicate.
2. Subject-predicate-object triples from a scene graph dataset.
3. The frequency of each triple in the training data.

Your task is to write structured, category-agnostic visual descriptions for the predicate.

Predicate:
{predicate}

Observed triples with frequency:
{triples}

Instructions:
- Use the triples and frequencies only to infer common visual regularities of the predicate.
- Do not mention any specific subject or object category names from the triples.
- Do not copy or restate the triples.
- Do not make frequent triples sound more correct than rare triples.
- Describe what can be visually observed in an image.
- Each field must be exactly one sentence.
- Keep each sentence concise and discriminative.
- If a factor is weak or usually irrelevant, still describe why it is weak in visual terms.
- Avoid dataset-specific language.

Output valid JSON only, with exactly the following fields:

{{
  "predicate": "{predicate}",
  "spatial_factor": "",
  "interaction_factor": "",
  "attribute_factor": "",
  "visual_definition": "",
  "negative_cues": ""
}}

Field definitions:
- spatial_factor: Describe the relative position, geometry, distance, orientation, containment, support, or contact pattern between the two entities.
- interaction_factor: Describe the physical action, functional use, manipulation, support, motion, attention, or intentional relation between the two entities.
- attribute_factor: Describe visible surface, material, texture, part-whole, ownership, affordance, attachment, or appearance cues related to the predicate.
- visual_definition: Give a general visual definition of the predicate as it should be recognized in an image.
- negative_cues: Describe visually similar cases where this predicate should not be used.
"""


REQUIRED_KEYS = [
    "predicate",
    "spatial_factor",
    "interaction_factor",
    "attribute_factor",
    "visual_definition",
    "negative_cues",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate LLM factor descriptions for predicate triplet statistics."
    )
    parser.add_argument("--input-json", required=True, help="Predicate triplet statistics JSON.")
    parser.add_argument("--output-json", required=True, help="Output description JSON.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--top-k", type=int, default=30, help="Most frequent triples per predicate.")
    parser.add_argument("--tail-k", type=int, default=10, help="Least frequent triples per predicate.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep between calls.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def save_json(path, data):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, path)


def parse_pair_key(key):
    text = str(key).strip()
    if " - " in text:
        subject, obj = text.split(" - ", 1)
        return subject.strip(), obj.strip()

    try:
        value = ast.literal_eval(text)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return str(value[0]), str(value[1])
    except (SyntaxError, ValueError):
        pass

    raise ValueError("Expected pair key like 'subject - object', got: {}".format(key))


def normalize_triples(raw_triples):
    if not isinstance(raw_triples, dict):
        raise ValueError("Expected predicate value to be a pair-count dict.")
    return [
        (*parse_pair_key(pair_key), int(count))
        for pair_key, count in raw_triples.items()
    ]


def select_triples(triples, top_k, tail_k):
    # Input JSON is expected to be frequency-sorted. Keep that order for top-k.
    selected = triples[:top_k]
    if tail_k > 0 and len(triples) > top_k:
        selected.extend(triples[-tail_k:])

    return selected


def format_triples(predicate, triples):
    if not triples:
        return "No observed triples are available."
    return "\n".join(
        "{}-{}-{}: {}".format(subject, predicate, obj, count)
        for subject, obj, count in triples
    )


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model response does not contain JSON: {}".format(text[:200]))
    return json.loads(text[start:end + 1])


def validate_result(predicate, result):
    missing = [key for key in REQUIRED_KEYS if key not in result]
    if missing:
        raise ValueError("Missing keys for {}: {}".format(predicate, missing))
    result = {key: str(result[key]).strip() for key in REQUIRED_KEYS}
    result["predicate"] = predicate
    return result


def call_openai(client, model, prompt):
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=0.2,
        )
        return response.output_text
    except AttributeError:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content


def generate_one(client, model, predicate, triples, max_retries):
    prompt = PROMPT_TEMPLATE.format(
        predicate=predicate,
        triples=format_triples(predicate, triples),
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            text = call_openai(client, model, prompt)
            return validate_result(predicate, extract_json(text))
        except Exception as error:
            last_error = error
            time.sleep(2 ** attempt)
    raise RuntimeError("Failed to generate {}: {}".format(predicate, last_error))


def main():
    args = parse_args()
    stats = load_json(args.input_json)
    outputs = OrderedDict()
    if os.path.exists(args.output_json) and not args.overwrite:
        outputs.update(load_json(args.output_json))

    if args.dry_run:
        for predicate, raw_triples in stats.items():
            triples = select_triples(normalize_triples(raw_triples), args.top_k, args.tail_k)
            prompt = PROMPT_TEMPLATE.format(
                predicate=predicate,
                triples=format_triples(predicate, triples),
            )
            print(prompt)
            break
        return

    if not args.api_key:
        raise RuntimeError("OPENAI_API_KEY is required. Set it or pass --api-key.")

    from openai import OpenAI

    client_kwargs = {"api_key": args.api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    for idx, (predicate, raw_triples) in enumerate(stats.items(), 1):
        if predicate in outputs and not args.overwrite:
            print("[{}/{}] skip {}".format(idx, len(stats), predicate))
            continue

        triples = select_triples(normalize_triples(raw_triples), args.top_k, args.tail_k)
        print("[{}/{}] generate {} with {} triples".format(
            idx, len(stats), predicate, len(triples)
        ))
        outputs[predicate] = generate_one(
            client=client,
            model=args.model,
            predicate=predicate,
            triples=triples,
            max_retries=args.max_retries,
        )
        save_json(args.output_json, outputs)
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
