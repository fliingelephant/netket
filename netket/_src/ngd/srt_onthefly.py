from collections.abc import Callable
from functools import partial

from einops import rearrange

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from jax.sharding import NamedSharding, PartitionSpec as P

from netket import jax as nkjax
from netket import config
from netket.jax._jacobian.default_mode import JacobianMode
from netket.utils import timing
from netket.utils.types import Array

from netket.jax import _ntk as nt


@timing.timed
@partial(
    jax.jit,
    static_argnames=(
        "log_psi",
        "solver_fn",
        "chunk_size",
        "mode",
        "moment_adaptive",
    ),
)
def srt_onthefly(
    log_psi,
    local_energies,
    parameters,
    model_state,
    samples,
    *,
    diag_shift: float | Array,
    solver_fn: Callable[[Array, Array], Array],
    mode: JacobianMode,
    proj_reg: float | Array | None = None,
    momentum: float | Array | None = None,
    old_updates: Array | None = None,
    moment_adaptive: bool = False,
    beta: float | Array = 0.995,
    v: Array | None = None,
    prev_updates: Array | None = None,
    chunk_size: int | None = None,
    weights: Array | None = None,
):
    if weights is not None:
        raise NotImplementedError(
            "Weighted samples are not yet supported in srt_onthefly."
        )

    N_mc = local_energies.size

    # Split all parameters into real and imaginary parts separately
    parameters_real, rss = nkjax.tree_to_real(parameters)

    # complex: (Nmc) -> (Nmc,2) - splitting real and imaginary output like 2 classes
    # real:    (Nmc) -> (Nmc,)  - no splitting
    def _apply_fn(parameters_real, samples, model_state):
        variables = {"params": rss(parameters_real), **model_state}
        log_amp = log_psi(variables, samples)

        if mode == "complex":
            re, im = log_amp.real, log_amp.imag
            return jnp.concatenate(
                (re[:, None], im[:, None]), axis=-1
            )  # shape [N_mc,2]
        else:
            return log_amp.real  # shape [N_mc, ]

    def jvp_f_chunk(parameters, model_state, vector, samples):
        r"""
        Creates the jvp of the function `_apply_fn` with respect to the parameters.
        This jvp is then evaluated in chunks of `chunk_size` samples.
        """
        f = lambda params: _apply_fn(params, samples, model_state=model_state)
        _, acc = jax.jvp(f, (parameters,), (vector,))
        return acc

    # MARCH (Gu et al. 2025, Eq. S17-S23): reparameterize via column-scaling
    #   θ̃ = D^(1/4) θ,  d/dθ̃ log ψ = J D^(-1/4) = U
    # We wrap _apply_fn so that differentiating wrt the scaled parameters gives U.
    # The linearization point is params_s_in = D^(1/4) params_real (same log ψ value).
    if moment_adaptive:
        if v is None:
            v = tree_map(jnp.ones_like, parameters_real)
        if prev_updates is None:
            prev_updates = tree_map(jnp.zeros_like, parameters_real)
        v_quart_inv = tree_map(
            lambda vv: jnp.power(vv, -0.25).astype(vv.dtype), v
        )

        def _apply_fn_scaled(params_s, samples, model_state):
            params_r = tree_map(lambda ps, vq: ps * vq, params_s, v_quart_inv)
            return _apply_fn(params_r, samples, model_state)

        # Linearization point θ̃ = D^(1/4) θ = θ / v_quart_inv.
        params_s_in = tree_map(lambda p, vq: p / vq, parameters_real, v_quart_inv)
    else:
        v_quart_inv = None
        _apply_fn_scaled = _apply_fn
        params_s_in = parameters_real

    # compute rhs of the linear system
    local_energies = local_energies.flatten()
    de = local_energies - jnp.mean(local_energies)

    # At the moment the final vjp is centered by centering the auxiliary vector a.
    # This is the same as centering the jacobian but may have larger variance.
    dv = 2.0 * de / jnp.sqrt(N_mc)  # shape [N_mc,]
    if mode == "complex":
        dv = jnp.stack([jnp.real(dv), jnp.imag(dv)], axis=-1)  # shape [N_mc,2]
    else:
        dv = jnp.real(dv)  # shape [N_mc,]

    # Momentum step (SPRING): ζ = dv - Õ φ, with φ = μ old_updates in raw coords.
    # Uses UNSCALED _apply_fn and parameters_real so the JVP is Õ @ old_updates.
    if momentum is not None:
        if old_updates is None:
            old_updates = tree_map(jnp.zeros_like, parameters_real)
        else:
            acc = nkjax.apply_chunked(
                jvp_f_chunk, in_axes=(None, None, None, 0), chunk_size=chunk_size
            )(parameters_real, model_state, old_updates, samples)

            avg = jnp.mean(acc, axis=0)
            acc = (acc - avg) / jnp.sqrt(N_mc)
            dv -= momentum * acc

    if mode == "complex":
        dv = jax.lax.collapse(dv, 0, 2)  # shape [2*N_mc,] or [N_mc, ] if not complex

    # Collect all samples on all MPI ranks, those label the columns of the T matrix
    all_samples = samples
    if config.netket_experimental_sharding:
        samples = jax.lax.with_sharding_constraint(
            samples, NamedSharding(jax.sharding.get_abstract_mesh(), P("S", None))
        )
        all_samples = jax.lax.with_sharding_constraint(
            samples, NamedSharding(jax.sharding.get_abstract_mesh(), P())
        )

    # NTK built from the SCALED apply (if moment_adaptive). This gives U U^T.
    _jacobian_contraction = nt.empirical_ntk_by_jacobian(
        f=_apply_fn_scaled,
        trace_axes=(),
        vmap_axes=0,
    )

    def jacobian_contraction(samples, all_samples, params_s, model_state):
        if config.netket_experimental_sharding:
            params_s = nkjax.lax.pcast(params_s, "S", to="varying")
        if chunk_size is None:
            # STRUCTURED_DERIVATIVES returns a complex array, but the imaginary part is zero
            # shape [N_mc/p.size, N_mc, 2, 2]
            return _jacobian_contraction(
                samples, all_samples, params_s, model_state=model_state
            ).real
        else:
            _all_samples, _ = nkjax.chunk(all_samples, chunk_size=chunk_size)
            ntk_local = jax.lax.map(
                lambda batch_lattice: _jacobian_contraction(
                    samples, batch_lattice, params_s, model_state=model_state
                ).real,
                _all_samples,
            )
            if mode == "complex":
                return rearrange(ntk_local, "nbatches i j z w -> i (nbatches j) z w")
            else:
                return rearrange(ntk_local, "nbatches i j -> i (nbatches j)")

    # If we are sharding, use shard_map manually
    if config.netket_experimental_sharding:
        mesh = jax.sharding.get_abstract_mesh()
        # SAMPLES, ALL_SAMPLES PARAMETERS_REAL
        in_specs = (P("S", None), P(), P(), P())
        out_specs = P("S", None)

        # By default, I'm not sure whether the jacobian_contraction of NeuralTangents
        # Is correctly automatically sharded across devices. So we force it to be
        # sharded with shard map to be sure

        jacobian_contraction = jax.shard_map(
            jacobian_contraction,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
        )

    # This disables the nkjax.sharding_decorator in here, which might appear
    # in the apply function inside.
    with nkjax.sharding._increase_SHARD_MAP_STACK_LEVEL():
        ntk_local = jacobian_contraction(
            samples, all_samples, params_s_in, model_state
        ).real

    # shape [N_mc, N_mc, 2, 2] or [N_mc, N_mc]
    if config.netket_experimental_sharding:
        # this sharding constraint should be useless, but let's keep it for safety.
        ntk = jax.lax.with_sharding_constraint(
            ntk_local, NamedSharding(jax.sharding.get_abstract_mesh(), P("S", None))
        )
    else:
        ntk = ntk_local
    if mode == "complex":
        # shape [2*N_mc, 2*N_mc] checked with direct calculation of J^T J
        ntk = rearrange(ntk, "i j z w -> (i z) (j w)")

    # Center the NTK by avoiding the construction of a big dense matrix to lower memory pressure.
    # Equivalent to the 'old'  delta = jnp.eye(N_mc) - 1 / N_mc
    # ntk = (delta_conc @ (ntk @ delta_conc)) / N_mc
    if mode == "complex":
        ntk = ntk.reshape(N_mc, 2, N_mc, 2)
        col_means = ntk.mean(axis=0, keepdims=True)  # all-reduce
        row_means = ntk.mean(axis=2, keepdims=True)  # all-reduce
        global_mean = col_means.mean(axis=2, keepdims=True)  # local, reuse mean_0
        ntk = ntk - col_means - row_means + global_mean
        ntk = ntk.reshape(2 * N_mc, 2 * N_mc)
    else:
        row_means = ntk.mean(axis=1, keepdims=True)  # local: mean over columns
        col_means = ntk.mean(axis=0, keepdims=True)  # all-reduce: mean over rows
        global_mean = col_means.mean()  # local: col_means is already replicated
        ntk = ntk - col_means - row_means + global_mean

    ntk = ntk / N_mc

    # Create identity matrix with same sharding as ntk: P("S", None)
    if config.netket_experimental_sharding:
        local_size = ntk.shape[0]
        identity = jnp.eye(local_size)
        identity = jax.lax.with_sharding_constraint(
            identity, NamedSharding(jax.sharding.get_abstract_mesh(), P("S", None))
        )
    else:
        identity = jnp.eye(ntk.shape[0])

    # add diag shift
    ntk_shifted = ntk + diag_shift * identity

    # add projection regularization
    if proj_reg is not None:
        ntk_shifted = ntk_shifted + proj_reg / N_mc

    # some solvers return a tuple, some others do not.
    aus_vector = solver_fn(ntk_shifted, dv)

    if isinstance(aus_vector, tuple):
        aus_vector, info = aus_vector
        if info is None:
            info = {}
    else:
        info = {}

    if config.netket_experimental_sharding:
        aus_vector = jax.lax.with_sharding_constraint(
            aus_vector,
            NamedSharding(jax.sharding.get_abstract_mesh(), P("S")),
        )

    aus_vector = jnp.squeeze(aus_vector)
    if mode == "complex":
        aus_vector = aus_vector.reshape((N_mc, 2))

    # Center the vector, equivalent to centering the Jacobian
    # This is equivalent to: aus_vector = delta_conc @ aus_vector
    aus_vector = (aus_vector - jnp.mean(aus_vector, axis=0, keepdims=True)) / jnp.sqrt(
        N_mc
    )
    # shape [N_mc // p.size,2]
    if config.netket_experimental_sharding:
        aus_vector = jax.lax.with_sharding_constraint(
            aus_vector,
            NamedSharding(
                jax.sharding.get_abstract_mesh(),
                P("S", *(None,) * (aus_vector.ndim - 1)),
            ),
        )

    # VJP against the SCALED apply → updates_scaled = U^T aus_vector  (scaled coords)
    vjp_fun = nkjax.vjp_chunked(
        _apply_fn_scaled,
        params_s_in,
        samples,
        model_state,
        chunk_size=chunk_size,
        chunk_argnums=1,
        nondiff_argnums=(1, 2),
    )

    (updates_scaled,) = vjp_fun(aus_vector)  # pytree [N_params,] in scaled coords

    # Unscale: dθ_part = D^(-1/4) · updates_scaled  (paper Eq. S23's D^(-1/2) Õ^T aus part)
    if moment_adaptive:
        updates = tree_map(
            lambda u, vq: u * vq.astype(u.dtype), updates_scaled, v_quart_inv
        )
    else:
        updates = updates_scaled

    # Add momentum: dθ = dθ_part + φ
    if momentum is not None:
        updates = tree_map(lambda x, y: x + momentum * y, updates, old_updates)
        old_updates = updates

    # MARCH second-moment EMA: v_k = β v_{k-1} + (dθ_k - dθ_{k-1})²  (pytree-wise)
    if moment_adaptive:
        diff = tree_map(lambda u, pu: u - pu, updates, prev_updates)
        new_v = tree_map(
            lambda vv, d: beta * vv + (d * d).astype(vv.dtype), v, diff
        )
        new_prev_updates = updates
    else:
        new_v = v
        new_prev_updates = prev_updates

    return rss(updates), old_updates, new_v, new_prev_updates, info
