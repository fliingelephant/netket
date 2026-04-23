from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from netket import jax as nkjax
from netket import config
from netket.utils import timing
from netket.utils.types import Array


@timing.timed
@partial(
    jax.jit,
    static_argnames=(
        "solver_fn",
        "mode",
        "moment_adaptive",
    ),
)
def _compute_srt_update(
    O_L,
    dv,
    *,
    diag_shift: float | Array,
    solver_fn: Callable[[Array, Array], Array],
    mode: str,
    proj_reg: float | Array | None = None,
    momentum: float | Array | None = None,
    old_updates: Array | None = None,
    moment_adaptive: bool = False,
    beta: float | Array = 0.995,
    v: Array | None = None,
    prev_updates: Array | None = None,
    params_structure,
):
    # Momentum step (SPRING): ζ = dv - Õ φ with φ = μ dθ_{k-1} in raw coords.
    # Uses the original (unscaled) Jacobian since φ and old_updates live in raw space.
    if momentum is not None:
        dv = dv - momentum * (O_L @ old_updates)

    # MARCH (Gu et al. 2025, Eq. S17-S23): reparameterize via column-scaling
    #   U = O_L · diag(v)^(-1/4),   NTK becomes U U^T = Õ D^(-1/2) Õ^T
    if moment_adaptive:
        v_quart_inv = jnp.power(v.astype(O_L.real.dtype), -0.25)
        O_L_eff = O_L * v_quart_inv[None, :]
    else:
        v_quart_inv = None
        O_L_eff = O_L

    # Here we reshard the jacobian from being sharded along the sample axis to being
    # sharded along the parameter axis, which we pad
    # (#ns, np) -> (ns, #np)
    # In theory jax could figure this out, but maybe he does not, so we do it by hand.
    O_LT = O_L_eff
    if config.netket_experimental_sharding:
        O_LT = nkjax.sharding.pad_axis_for_sharding(O_LT, axis=1, padding_value=0.0)
        O_LT = jax.lax.with_sharding_constraint(
            O_LT,
            NamedSharding(jax.sharding.get_abstract_mesh(), P(None, "S")),
        )
        dv = jax.lax.with_sharding_constraint(
            dv, NamedSharding(jax.sharding.get_abstract_mesh(), P())
        )

    # This does the contraction (ns, #np) x (#np, ns) -> (ns, ns).
    # When using sharding the sum over #ns is done automatically.
    # When using MPI we need to do it manually with an allreduce_sum.
    matrix = O_LT @ O_LT.T
    matrix_side = matrix.shape[-1]  # * it can be ns or 2*ns, depending on mode

    shifted_matrix = jax.lax.add(
        matrix, diag_shift * jnp.eye(matrix_side, dtype=matrix.dtype)
    )
    # replicate

    if proj_reg is not None:
        shifted_matrix = jax.lax.add(
            shifted_matrix, jnp.full_like(shifted_matrix, proj_reg / matrix_side)
        )

    aus_vector = solver_fn(shifted_matrix, dv)

    # Some solvers return a tuple, some others do not.
    if isinstance(aus_vector, tuple):
        aus_vector, info = aus_vector
        if info is None:
            info = {}
    else:
        info = {}

    # dθ = D^(-1/2) · (O_L^T aus) + φ  (paper Eq. S23).
    # Uses the raw O_L so O_L_eff can be freed right after the matrix-form step.
    if moment_adaptive:
        updates = (v_quart_inv * v_quart_inv) * (O_L.T @ aus_vector)
    else:
        updates = O_L.T @ aus_vector
    if momentum is not None:
        updates = updates + momentum * old_updates
        new_old_updates = updates
    else:
        new_old_updates = old_updates

    # MARCH second-moment EMA: v_k = β v_{k-1} + (dθ_k - dθ_{k-1})²
    # Done in flat real-valued space, BEFORE the complex repack below.
    if moment_adaptive:
        diff = updates - prev_updates
        if jnp.iscomplexobj(diff):
            diff_sq = (diff * diff.conj()).real
        else:
            diff_sq = diff * diff
        new_v = beta * v + diff_sq.astype(v.dtype)
        new_prev_updates = updates
    else:
        new_v = v
        new_prev_updates = prev_updates

    # If complex mode and we have complex parameters, we need
    # To repack the real coefficients in order to get complex updates
    if mode == "complex" and nkjax.tree_leaf_iscomplex(params_structure):
        num_p = updates.shape[-1] // 2
        updates = updates[:num_p] + 1j * updates[num_p:]

    if config.netket_experimental_sharding:
        out_shardings = NamedSharding(
            jax.sharding.get_abstract_mesh(), P(*(None,) * updates.ndim)
        )
        updates = jax.lax.with_sharding_constraint(updates, out_shardings)

    return updates, new_old_updates, new_v, new_prev_updates, info
