"""Robot-neutral semantic geometry derived from a loaded MuJoCo model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from rimkit.kinematics import position, rotation, transform


class BodyTransformSource(Protocol):
    """Minimum model-adapter surface required for neutral JOI extraction."""

    def get_body_transform(self, name: str) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class NeutralRobotGeometry:
    """Neutral semantic transforms and lengths used to scale source limbs."""

    body_transforms: Mapping[str, NDArray[np.float64]]
    length_transforms: Mapping[str, NDArray[np.float64]]
    link_lengths: Mapping[str, float]


def _readonly_transform_map(
    values: Mapping[str, NDArray[np.float64]],
) -> Mapping[str, NDArray[np.float64]]:
    copied: dict[str, NDArray[np.float64]] = {}
    for key, value in values.items():
        item = np.array(value, dtype=np.float64, copy=True)
        item.setflags(write=False)
        copied[key] = item
    return MappingProxyType(copied)


def semantic_body_transforms(
    model: BodyTransformSource,
    joi_bodies: Mapping[str, str],
    *,
    base_between_hips: bool,
) -> Mapping[str, NDArray[np.float64]]:
    """Resolve body JOIs plus the virtual neck and optional mid-hip."""

    transforms = {key: model.get_body_transform(body_name) for key, body_name in joi_bodies.items()}
    neck_position = 0.5 * (position(transforms["rs"]) + position(transforms["ls"]))
    transforms["neck"] = transform(neck_position, np.eye(3, dtype=np.float64))
    if base_between_hips:
        base_position = 0.5 * (position(transforms["rp"]) + position(transforms["lp"]))
        transforms["base"] = transform(base_position, rotation(transforms["base"]))
    return _readonly_transform_map(transforms)


def derive_neutral_geometry(
    model: BodyTransformSource,
    joi_bodies: Mapping[str, str],
    *,
    link_length_base_reference: str,
) -> NeutralRobotGeometry:
    """Compute the exact neutral link lengths used by the research DMR path."""

    body = semantic_body_transforms(model, joi_bodies, base_between_hips=False)
    lengths = semantic_body_transforms(
        model,
        joi_bodies,
        base_between_hips=link_length_base_reference == "legacy_midhip",
    )

    pairs = {
        "base_spine": ("base", "spine"),
        "spine_neck": ("spine", "neck"),
        "neck_rs": ("neck", "rs"),
        "rs_re": ("rs", "re"),
        "re_rw": ("re", "rw"),
        "neck_ls": ("neck", "ls"),
        "ls_le": ("ls", "le"),
        "le_lw": ("le", "lw"),
        "base_rp": ("base", "rp"),
        "rp_rk": ("rp", "rk"),
        "rk_ra": ("rk", "ra"),
        "ra_rt": ("ra", "rt"),
        "base_lp": ("base", "lp"),
        "lp_lk": ("lp", "lk"),
        "lk_la": ("lk", "la"),
        "la_lt": ("la", "lt"),
    }
    link_lengths = {
        key: float(np.linalg.norm(position(lengths[end]) - position(lengths[start])))
        for key, (start, end) in pairs.items()
    }
    invalid = sorted(
        key for key, value in link_lengths.items() if not np.isfinite(value) or value <= 0.0
    )
    if invalid:
        raise ValueError("Robot neutral geometry has invalid link lengths: " + ", ".join(invalid))

    return NeutralRobotGeometry(
        body_transforms=body,
        length_transforms=lengths,
        link_lengths=MappingProxyType(link_lengths),
    )


__all__ = [
    "BodyTransformSource",
    "NeutralRobotGeometry",
    "derive_neutral_geometry",
    "semantic_body_transforms",
]
