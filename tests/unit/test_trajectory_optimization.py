from __future__ import annotations

import numpy as np
import pytest

from core_retarget.optimization import same_length_jerk_matrix, shape_trajectory_1d


def test_same_length_jerk_matrix_preserves_legacy_boundary_rows() -> None:
    matrix = same_length_jerk_matrix(6, 0.5)

    expected = np.asarray(
        [
            [-8.0, 24.0, -24.0, 8.0, 0.0, 0.0],
            [0.0, -8.0, 24.0, -24.0, 8.0, 0.0],
            [4.0, -8.0, 0.0, 8.0, -4.0, 0.0],
            [0.0, 4.0, -8.0, 0.0, 8.0, -4.0],
            [0.0, 8.0, -24.0, 24.0, -8.0, 0.0],
            [0.0, 0.0, 8.0, -24.0, 24.0, -8.0],
        ]
    )
    np.testing.assert_array_equal(matrix, expected)


def test_four_sample_jerk_matrix_uses_legacy_one_sided_rows() -> None:
    matrix = same_length_jerk_matrix(4, 1.0)

    np.testing.assert_array_equal(
        matrix,
        np.asarray(
            [
                [-1.0, 3.0, -3.0, 1.0],
                [-1.0, 3.0, -3.0, 1.0],
                [1.0, -3.0, 3.0, -1.0],
                [1.0, -3.0, 3.0, -1.0],
            ]
        ),
    )


def test_short_jerk_matrix_matches_repeated_first_difference() -> None:
    matrix = same_length_jerk_matrix(3, 1.0)
    first = np.asarray(
        [
            [-1.0, 1.0, 0.0],
            [-0.5, 0.0, 0.5],
            [0.0, -1.0, 1.0],
        ]
    )

    np.testing.assert_array_equal(matrix, first @ first @ first)


@pytest.mark.parametrize("sample_count", [0, -1])
def test_jerk_matrix_rejects_nonpositive_sample_count(sample_count: int) -> None:
    with pytest.raises(ValueError, match="sample_count"):
        same_length_jerk_matrix(sample_count, 0.1)


@pytest.mark.parametrize("sample_count", [True, 3.5])
def test_jerk_matrix_rejects_noninteger_sample_count(sample_count: object) -> None:
    with pytest.raises(TypeError, match="sample_count"):
        same_length_jerk_matrix(sample_count, 0.1)  # type: ignore[arg-type]


@pytest.mark.parametrize("sample_period", [0.0, -0.1, np.inf, np.nan])
def test_jerk_matrix_rejects_invalid_sample_period(sample_period: float) -> None:
    with pytest.raises(ValueError, match="sample_period"):
        same_length_jerk_matrix(5, sample_period)


def test_jerk_matrix_is_read_only() -> None:
    matrix = same_length_jerk_matrix(6, 0.1)

    assert not matrix.flags.writeable
    with pytest.raises(ValueError):
        matrix[0, 0] = 0.0


@pytest.mark.parametrize("tracking_norm", [1, 2])
def test_shape_trajectory_supports_both_legacy_tracking_norms(tracking_norm: int) -> None:
    seconds = np.arange(12, dtype=np.float64) / 30.0
    reference = np.linspace(-0.2, 0.4, len(seconds))

    result = shape_trajectory_1d(
        seconds,
        reference,
        tracking_norm=tracking_norm,
        jerk_weight=1e-5,
    )

    np.testing.assert_allclose(result.values, reference, rtol=0.0, atol=2e-6)
    assert result.status in ("optimal", "optimal_inaccurate")
    assert result.solver == "CLARABEL"
    assert result.z is result.values
    assert not result.values.flags.writeable
    with pytest.raises(ValueError):
        result.values[0] = 1.0


def test_shape_trajectory_reduces_same_length_jerk() -> None:
    seconds = np.arange(16, dtype=np.float64) / 30.0
    reference = np.asarray([0.0, 1.0] * 8, dtype=np.float64)
    matrix = same_length_jerk_matrix(len(seconds), 1.0 / 30.0)

    result = shape_trajectory_1d(seconds, reference)

    assert np.linalg.norm(matrix @ result.values) < np.linalg.norm(matrix @ reference)


def test_unknown_preferred_solver_falls_back_to_clarabel() -> None:
    seconds = np.arange(6, dtype=np.float64) * 0.1
    result = shape_trajectory_1d(
        seconds,
        np.zeros_like(seconds),
        preferred_solver="not_a_solver",
    )

    assert result.solver == "CLARABEL"


@pytest.mark.parametrize(
    ("seconds", "reference", "error"),
    [
        ([0.0], [1.0], "at least two"),
        ([0.0, 0.1], [1.0], "same shape"),
        ([0.0, np.nan], [1.0, 2.0], "finite"),
        ([0.0, 0.0], [1.0, 2.0], "strictly increasing"),
        ([0.0, 0.1, 0.3], [1.0, 2.0, 3.0], "uniformly sampled"),
        ([0.0, 0.1], [1.0, np.inf], "reference"),
    ],
)
def test_shape_trajectory_validates_samples(
    seconds: list[float],
    reference: list[float],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        shape_trajectory_1d(seconds, reference)


@pytest.mark.parametrize(
    ("tracking_norm", "jerk_weight", "error"),
    [
        (3, 1e-5, "tracking_norm"),
        (1, -1.0, "jerk_weight"),
        (1, np.inf, "jerk_weight"),
    ],
)
def test_shape_trajectory_validates_objective_settings(
    tracking_norm: int,
    jerk_weight: float,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        shape_trajectory_1d(
            [0.0, 0.1],
            [0.0, 0.0],
            tracking_norm=tracking_norm,
            jerk_weight=jerk_weight,
        )
