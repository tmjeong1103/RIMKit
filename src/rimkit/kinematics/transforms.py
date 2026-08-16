"""Small rigid-transform helpers shared by retargeting stages."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def transform(
    position: ArrayLike,
    rotation: ArrayLike,
) -> NDArray[np.float64]:
    """Build a homogeneous transform from a 3-vector and 3x3 rotation."""

    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    value[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return value


def position(value: ArrayLike) -> NDArray[np.float64]:
    """Return a copy of the translation component of a transform."""

    return np.asarray(value, dtype=np.float64).reshape(4, 4)[:3, 3].copy()


def rotation(value: ArrayLike) -> NDArray[np.float64]:
    """Return a copy of the rotation component of a transform."""

    return np.asarray(value, dtype=np.float64).reshape(4, 4)[:3, :3].copy()


def unit_vector(
    value: ArrayLike,
    *,
    fallback: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> NDArray[np.float64]:
    """Normalize a 3-vector with the DMR degeneracy fallback."""

    vector = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-10:
        return np.asarray(fallback, dtype=np.float64)
    return vector / norm


__all__ = ["position", "rotation", "transform", "unit_vector"]
