import logging
import torch
from tqdm import tqdm


__all__ = [
    "SDE_sampler_manifolds_OLLA_P",
    "SDE_sampler_manifolds_OLLA",
    "SDE_sampler_manifolds_ULLA_P",
    "SDE_sampler_manifolds_ULLA",
]


def _gamma_tensor(kwargs, *, device: torch.device):
    gamma = float(kwargs.get("gamma", 100.0))
    if "gamma" not in kwargs:
        logging.info("gamma not specified; defaulting to 100.0")
    return torch.as_tensor(gamma, device=device)


def _timesteps(sde, init, reverse):
    start, end = (sde.T, 0.0) if reverse else (0.0, sde.T)
    return torch.linspace(start, end, sde.N + 1, device=init.device).reshape(-1, 1).repeat(1, init.shape[0])


def _sample_velocity(manifold, x, x_back, sigma_back, delta_t, initial_step, init_v=None):
    if init_v is not None and initial_step:
        return init_v
    if initial_step:
        return manifold.project_onto_tangent_space(torch.randn_like(x), base_point=x)
    v = (x - x_back) / (sigma_back ** 2 * delta_t)
    return manifold.project_onto_tangent_space(v, base_point=x)


def _trajectory_result(x, x_hist, converged):
    other = {"converged_traj": converged, "x_hist_all": x_hist}
    return x[converged], x_hist[:, converged], other


def _update_convergence(converged, step_converged):
    return converged & step_converged.bool()


@torch.no_grad()
def SDE_sampler_manifolds_OLLA_P(
    sde,
    manifold,
    init,
    reverse,
    score_net=None,
    keep_quiet=False,
    **kwargs,
):
    """Overdamped Langevin sampler with projection after each step."""
    device = init.device
    timesteps = _timesteps(sde, init, reverse)
    drift_diffusion_fn = sde.reverse(score_net).sde if reverse else sde.sde

    x_hist = torch.zeros(sde.N + 1, *init.shape, device=device)
    x = init
    x_hist[0] = x.clone()
    converged = torch.ones(init.shape[0], dtype=torch.bool, device=device)

    for i in tqdm(range(sde.N), mininterval=2.0, disable=keep_quiet):
        t = timesteps[i]
        delta_t = (timesteps[i + 1] - timesteps[i]).reshape(-1, 1)
        z = torch.randn_like(x)
        if reverse:
            drift, diffusion = drift_diffusion_fn(x, score_t=t, diff_t=t + delta_t.flatten())
        else:
            drift, diffusion = drift_diffusion_fn(x, t)

        diffusion = diffusion.reshape(-1, 1)
        tangent_vec = manifold.project_onto_tangent_space(
            drift * delta_t / 2 + diffusion * torch.sqrt(torch.abs(delta_t)) * z,
            base_point=x,
        )
        x, step_converged = manifold.project_onto_manifold_with_base(tangent_vec, base_point=x)
        converged = _update_convergence(converged, step_converged)
        x_hist[i + 1] = x.clone()

    if not keep_quiet:
        logging.info(f"{init.shape[0] - converged.sum()} of {init.shape[0]} trajectories are dropped.")
    return _trajectory_result(x, x_hist, converged)


@torch.no_grad()
def SDE_sampler_manifolds_OLLA(
    sde,
    manifold,
    init,
    reverse,
    score_net=None,
    keep_quiet=False,
    **kwargs,
):
    """Overdamped Langevin sampler with decaying constraint correction."""
    device = init.device
    alpha = kwargs.get("alpha", 100.0)
    projection_mode = kwargs.get("projection_mode", "implicit")
    if projection_mode not in ("implicit", "explicit"):
        logging.info(f"Unknown projection_mode '{projection_mode}' for OLLA; defaulting to implicit.")
        projection_mode = "implicit"

    timesteps = _timesteps(sde, init, reverse)
    drift_diffusion_fn = sde.reverse(score_net).sde if reverse else sde.sde

    x_hist = torch.zeros(sde.N + 1, *init.shape, device=device)
    x = init
    x_hist[0] = x.clone()
    converged = torch.ones(init.shape[0], dtype=torch.bool, device=device)

    for i in tqdm(range(sde.N), mininterval=2.0, disable=keep_quiet):
        t = timesteps[i]
        delta_t = (timesteps[i + 1] - timesteps[i]).reshape(-1, 1)
        z = torch.randn_like(x)
        if reverse:
            drift, diffusion = drift_diffusion_fn(x, score_t=t, diff_t=t + delta_t.flatten())
        else:
            drift, diffusion = drift_diffusion_fn(x, t)

        diffusion = diffusion.reshape(-1, 1)
        tangent_vec = manifold.project_onto_tangent_space(
            drift * delta_t / 2 + diffusion * torch.sqrt(torch.abs(delta_t)) * z,
            base_point=x,
        )

        if projection_mode == "implicit":
            x = manifold.adding_correction_decaying_implicit(
                base_point=x + tangent_vec,
                delta_t=delta_t,
                alpha=alpha,
                sigma_sq=diffusion ** 2,
            )
        else:
            x = manifold.adding_correction_decaying(
                tangent_vec,
                base_point=x,
                delta_t=delta_t,
                alpha=alpha,
                sigma_sq=diffusion ** 2,
            )

        step_converged = torch.ones(x.shape[0], dtype=torch.bool, device=device)
        if i == sde.N - 1:
            zero = torch.zeros_like(tangent_vec)
            x, step_converged = manifold.project_onto_manifold_with_base(zero, base_point=x)
        converged = _update_convergence(converged, step_converged)
        x_hist[i + 1] = x.clone()

    if not keep_quiet:
        logging.info(f"{init.shape[0] - converged.sum()} of {init.shape[0]} trajectories are dropped.")
    return _trajectory_result(x, x_hist, converged)


@torch.no_grad()
def SDE_sampler_manifolds_ULLA_P(
    sde,
    manifold,
    init,
    reverse,
    init_v=None,
    score_net=None,
    keep_quiet=False,
    **kwargs,
):
    """Underdamped Langevin sampler with projection after each position update."""
    device = init.device
    gamma = _gamma_tensor(kwargs, device=device)
    timesteps = _timesteps(sde, init, reverse)
    rsde = sde.reverse(score_net, underdamped=True) if reverse else None

    x_hist = torch.zeros(sde.N + 1, *init.shape, device=device)
    x = init
    x_hist[0] = x.clone()
    converged = torch.ones(init.shape[0], dtype=torch.bool, device=device)

    for i in tqdm(range(sde.N), mininterval=2.0, disable=keep_quiet):
        t = timesteps[i]
        delta_t = timesteps[i + 1] - timesteps[i]
        sigma = sde.get_diffusion(t + delta_t if reverse else t).reshape(-1, 1)
        sigma_back = sde.get_diffusion(t if reverse else t - delta_t).reshape(-1, 1)
        delta_t = delta_t.reshape(-1, 1)

        v = _sample_velocity(manifold, x, x if i == 0 else x_hist[i - 1], sigma_back, delta_t, i == 0, init_v)
        a = torch.exp(-(sigma ** 2) * torch.abs(delta_t) * gamma)
        sqrt_1ma2 = torch.sqrt(torch.abs(1.0 - a ** 2))
        z = torch.randn_like(x)

        if reverse:
            drift = rsde.drift_score(x, v, t) - sde.drift_b(x)
        else:
            drift = sde.drift_b(x)
        velocity = a * v + (sigma ** 2) * torch.abs(delta_t) * drift + sqrt_1ma2 * z
        step = (sigma ** 2) * delta_t * manifold.project_onto_tangent_space(velocity, base_point=x)
        x, step_converged = manifold.project_onto_manifold_with_base(step, base_point=x)
        converged = _update_convergence(converged, step_converged)
        x_hist[i + 1] = x.clone()

    if not keep_quiet:
        logging.info(f"{init.shape[0] - converged.sum()} of {init.shape[0]} trajectories are dropped.")
    return _trajectory_result(x, x_hist, converged)


@torch.no_grad()
def SDE_sampler_manifolds_ULLA(
    sde,
    manifold,
    init,
    reverse,
    init_v=None,
    score_net=None,
    keep_quiet=False,
    **kwargs,
):
    """Underdamped Langevin sampler with decaying constraint correction."""
    device = init.device
    alpha = kwargs.get("alpha", 100.0)
    terminal_projection = kwargs.get("terminal_projection", True)
    projection_threshold = kwargs.get("projection_threshold", 1e-5)
    projection_mode = kwargs.get("projection_mode", "explicit")
    if projection_mode not in ("implicit", "explicit"):
        logging.info(f"Unknown projection_mode '{projection_mode}' for ULLA; defaulting to explicit.")
        projection_mode = "explicit"

    gamma = _gamma_tensor(kwargs, device=device)
    timesteps = _timesteps(sde, init, reverse)
    rsde = sde.reverse(score_net, underdamped=True) if reverse else None

    x_hist = torch.zeros(sde.N + 1, *init.shape, device=device)
    x = init
    x_hist[0] = x.clone()
    converged = torch.ones(init.shape[0], dtype=torch.bool, device=device)

    for i in tqdm(range(sde.N), mininterval=2.0, disable=keep_quiet):
        t = timesteps[i]
        delta_t = timesteps[i + 1] - timesteps[i]
        sigma = sde.get_diffusion(t + delta_t if reverse else t).reshape(-1, 1)
        sigma_back = sde.get_diffusion(t if reverse else t - delta_t).reshape(-1, 1)
        delta_t = delta_t.reshape(-1, 1)

        v = _sample_velocity(manifold, x, x if i == 0 else x_hist[i - 1], sigma_back, delta_t, i == 0, init_v)
        a = torch.exp(-(sigma ** 2) * torch.abs(delta_t) * gamma)
        sqrt_1ma2 = torch.sqrt(torch.abs(1.0 - a ** 2))
        z = torch.randn_like(x)

        if reverse:
            drift = rsde.drift_score(x, v, t) - sde.drift_b(x)
        else:
            drift = sde.drift_b(x)
        velocity = a * v + (sigma ** 2) * torch.abs(delta_t) * drift + sqrt_1ma2 * z
        step = (sigma ** 2) * delta_t * manifold.project_onto_tangent_space(velocity, base_point=x)

        if projection_mode == "implicit":
            x = manifold.adding_correction_decaying_implicit(
                base_point=x + step,
                delta_t=delta_t,
                alpha=alpha,
                sigma_sq=sigma ** 2,
            )
        else:
            x = manifold.adding_correction_decaying(
                step,
                base_point=x,
                delta_t=delta_t,
                alpha=alpha,
                sigma_sq=sigma ** 2,
            )

        step_converged = torch.ones(x.shape[0], dtype=torch.bool, device=device)
        if terminal_projection and i == sde.N - 1:
            zero = torch.zeros_like(step)
            x, step_converged = manifold.project_onto_manifold_with_base(
                zero, base_point=x, threshold=projection_threshold
            )
        converged = _update_convergence(converged, step_converged)
        x_hist[i + 1] = x.clone()

    if not keep_quiet:
        logging.info(f"{init.shape[0] - converged.sum()} of {init.shape[0]} trajectories are dropped.")
    return _trajectory_result(x, x_hist, converged)
