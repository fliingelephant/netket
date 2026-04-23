from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from netket import jax as nkjax
from netket.utils.types import Array

from netket import config

from netket._src.ngd.kwargs import ensure_accepts_kwargs


@partial(
    jax.jit,
    static_argnames=(
        "solver_fn",
        "mode",
        "moment_adaptive",
    ),
)
def _compute_sr_update(
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
    # We concretize the solver function to ensure it accepts the additional argument `dv`.
    # Typically solvers only accept the matrix and the right-hand side.
    solver_fn = ensure_accepts_kwargs(solver_fn, "dv")

    if proj_reg is not None:
        raise ValueError("proj_reg not implemented for SR")

    # MARCH (Gu et al. 2025, Eq. S17-S23): reparameterize via column-scaling
    #   U = O_L · diag(v)^(-1/4),  dθ = diag(v)^(-1/4) · dθ̃ + φ
    # Reduces to SPRING when moment_adaptive=False (v_quart_inv ≡ 1).
    if moment_adaptive:
        v_quart_inv = jnp.power(v.astype(O_L.real.dtype), -0.25)
        O_L_eff = O_L * v_quart_inv[None, :]
    else:
        v_quart_inv = None
        O_L_eff = O_L

    # (np, #ns) x (#ns) -> (np) - where the sum over #ns is done automatically
    F = O_L_eff.T @ dv

    # Add momentum term F += λ μ dθ̃_{k-1}; in scaled coords dθ̃ = v^(1/4) · dθ.
    if momentum is not None:
        if moment_adaptive:
            F = F + diag_shift * momentum * old_updates / v_quart_inv
        else:
            F = F + diag_shift * momentum * old_updates

    # This does the contraction (np, #ns) x (#ns, np) -> (np, np).
    matrix = O_L_eff.T @ O_L_eff
    matrix_side = matrix.shape[-1]  # * it can be ns or 2*ns, depending on mode

    shifted_matrix = jax.lax.add(
        matrix, diag_shift * jnp.eye(matrix_side, dtype=matrix.dtype)
    )
    solve_out = solver_fn(shifted_matrix, F, dv=dv)

    # Some solvers return a tuple, some others do not.
    if isinstance(solve_out, tuple):
        pi_scaled, info = solve_out
        if info is None:
            info = {}
    else:
        pi_scaled = solve_out
        info = {}

    # Unscale: dθ = diag(v)^(-1/4) · dθ̃
    if moment_adaptive:
        updates = v_quart_inv * pi_scaled
    else:
        updates = pi_scaled

    if momentum is not None:
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
        updates = jax.lax.with_sharding_constraint(
            updates, NamedSharding(jax.sharding.get_abstract_mesh(), P())
        )

    return updates, new_old_updates, new_v, new_prev_updates, info
