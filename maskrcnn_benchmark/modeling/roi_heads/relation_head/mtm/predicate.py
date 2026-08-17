from dataclasses import dataclass


@dataclass(frozen=True)
class MTMPredicateSpec:
    """Dataset predicate vocabulary and its supported evaluation modes."""

    relation_names: tuple
    mode_indices: dict
    text_filter: object

    def __init__(self, relation_names, mode_indices, text_filter):
        object.__setattr__(self, "relation_names", tuple(relation_names))
        object.__setattr__(
            self,
            "mode_indices",
            {name: tuple(indices) for name, indices in mode_indices.items()},
        )
        object.__setattr__(self, "text_filter", text_filter)
        self._validate()

    def _validate(self):
        if not self.relation_names or self.relation_names[0] != "__background__":
            raise ValueError("MTM relation_names must start with __background__")
        relation_count = len(self.relation_names)
        for mode, indices in self.mode_indices.items():
            if not indices or indices[0] != 0:
                raise ValueError("MTM mode {} must start with background index 0".format(mode))
            if len(set(indices)) != len(indices):
                raise ValueError("MTM mode {} contains duplicate predicate indices".format(mode))
            if min(indices) < 0 or max(indices) >= relation_count:
                raise ValueError("MTM mode {} contains an out-of-range predicate index".format(mode))

    @property
    def base_indices(self):
        return self.mode_indices["base"]

    @property
    def novel_indices(self):
        return self.mode_indices["novel"]

    def resolve(self, mode):
        if mode not in self.mode_indices:
            raise ValueError("Unsupported MTM predicate mode: {}".format(mode))
        indices = self.mode_indices[mode]
        names = tuple(self.relation_names[index] for index in indices)
        inference_filter = self.text_filter.iloc[list(indices)]
        if inference_filter.index.tolist() != list(indices):
            raise ValueError("MTM predicate filter rows do not match mode {}".format(mode))
        return names, inference_filter

    def validate_names(self, mode, names):
        expected, _ = self.resolve(mode)
        if tuple(names) != expected:
            raise ValueError(
                "MTM {} predicate order mismatch: expected {}, got {}".format(
                    mode, list(expected), list(names)
                )
            )
