"""Analytic value/gradient checks for the shared cell-centered sampler."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from astar.esdf import sample_batched_esdf
from astar.esdf import sample_batched_esdf_with_valid


def test_final_interpolation_cells_preserve_bilinear_values_and_gradients():
    # A bilinear surface with a cross term distinguishes true interpolation
    # from nearest-neighbor lookup and from interpolation along just one axis.
    rows, cols = jnp.meshgrid(jnp.arange(3.0), jnp.arange(4.0), indexing="ij")
    field = (2.0 + 3.0 * cols - 2.0 * rows + 0.5 * cols * rows)[None]
    grid_xy = jnp.asarray([[[2.75, 0.25], [0.25, 1.75], [2.75, 1.75]]])
    offset = jnp.asarray([-2.0, 1.0])
    resolution = 0.25
    xy = offset + (grid_xy + 0.5) * resolution

    def sample(points):
        return sample_batched_esdf_with_valid(field, points, -2.0, 1.0, resolution)

    values, valid = jax.jit(sample)(xy)
    x, y = np.asarray(grid_xy[..., 0]), np.asarray(grid_xy[..., 1])
    np.testing.assert_allclose(values, 2.0 + 3.0 * x - 2.0 * y + 0.5 * x * y)
    assert np.all(valid)
    grad = jax.jit(jax.grad(lambda points: jnp.sum(sample(points)[0])))(xy)
    expected_grad = np.stack([3.0 + 0.5 * y, -2.0 + 0.5 * x], axis=-1) / resolution
    np.testing.assert_allclose(grad, expected_grad, atol=1e-6)


def test_edge_centers_corners_and_outside_have_correct_values_and_validity():
    field = jnp.asarray([[[1.0, 2.0, 4.0], [5.0, 7.0, 9.0]]])
    xy = jnp.asarray(
        [
            [
                [0.5, 0.5],
                [2.5, 0.5],
                [0.5, 1.5],
                [2.5, 1.5],
                [2.5, 1.0],
                [1.0, 1.5],
                [0.4, 0.5],
                [2.6, 1.5],
                [0.5, 0.4],
                [2.5, 1.6],
            ]
        ]
    )
    sampled, valid = sample_batched_esdf_with_valid(field, xy, 0.0, 0.0, 1.0)
    np.testing.assert_allclose(sampled, [[1.0, 4.0, 5.0, 9.0, 6.5, 6.0, 1.0, 9.0, 1.0, 9.0]])
    np.testing.assert_array_equal(valid, [[True] * 6 + [False] * 4])
    masked = sample_batched_esdf(field, xy, 0.0, 0.0, 1.0)
    assert np.all(np.isneginf(masked[:, 6:]))


def test_two_by_two_grids_use_each_maps_origin_and_resolution():
    field = jnp.asarray([[[0.0, 2.0], [4.0, 6.0]], [[10.0, 12.0], [14.0, 16.0]]])
    xy = jnp.asarray([[[1.25, 1.25]], [[-1.375, 3.625]]])
    sampled, valid = sample_batched_esdf_with_valid(
        field,
        xy,
        jnp.asarray([0.0, -2.0]),
        jnp.asarray([0.0, 3.0]),
        jnp.asarray([1.0, 0.5]),
    )
    np.testing.assert_allclose(sampled, [[4.5], [14.5]])
    assert np.all(valid)


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_coordinates_are_not_valid(coordinate):
    xy = jnp.asarray([[[coordinate, 1.0], [1.0, coordinate]]])
    masked = jax.jit(sample_batched_esdf)(jnp.ones((1, 2, 2)), xy, 0.0, 0.0, 1.0)
    assert np.all(np.isneginf(masked))
