"""Plain-YAML configuration helpers (no Hydra: fewer moving parts, easy to read)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .data import CATEGORIES, category_type


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML mapping.")
    return data


@dataclass(frozen=True)
class CategorySplit:
    dev: tuple[str, ...]
    held_out: tuple[str, ...]
    selection: str = "unspecified"

    def validate(self) -> CategorySplit:
        overlap = set(self.dev) & set(self.held_out)
        if overlap:
            raise ValueError(f"Categories in both dev and held-out: {sorted(overlap)}")
        listed = set(self.dev) | set(self.held_out)
        if listed != set(CATEGORIES):
            missing = sorted(set(CATEGORIES) - listed)
            unknown = sorted(listed - set(CATEGORIES))
            raise ValueError(
                f"Split must cover all 15 categories; missing={missing}, unknown={unknown}"
            )
        held_types = {category_type(c) for c in self.held_out}
        if held_types != {"texture", "object"}:
            raise ValueError("Held-out set must contain at least one texture and one object.")
        return self


def _flatten(group: Any, key: str) -> tuple[str, ...]:
    """Accept either a flat list or {textures: [...], objects: [...]} and type-check it."""
    if isinstance(group, list):
        return tuple(group)
    if not isinstance(group, dict):
        raise ValueError(f"'{key}' must be a list or a mapping of textures/objects.")
    out: list[str] = []
    for kind, singular in (("textures", "texture"), ("objects", "object")):
        for c in group.get(kind, []) or []:
            if category_type(c) != singular:
                raise ValueError(
                    f"{c!r} is listed under {key}.{kind} but has type {category_type(c)!r}."
                )
            out.append(c)
    return tuple(out)


def load_split(path: str | Path = "configs/split.yaml") -> CategorySplit:
    cfg = load_yaml(path)
    return CategorySplit(
        dev=_flatten(cfg["dev"], "dev"),
        held_out=_flatten(cfg["held_out"], "held_out"),
        selection=str(cfg.get("selection", "unspecified")),
    ).validate()
