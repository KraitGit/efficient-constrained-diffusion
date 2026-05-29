import torch
import math

def project_tangent(manifold, y, base, project_to_tangent=True):
    if not project_to_tangent:
        return y
    if y.ndim != 2:
        raise ValueError(f"Input y must have shape (batch, dim), got {y.shape}")
    if base.ndim != 2:
        raise ValueError(f"Input base must have shape (batch, dim), got {base.shape}")
    
    return manifold.project_onto_tangent_space(y, base_point=base)


def G_bwd_overdamped(manifold, x_prev, x_next, sigmas, dt, drift, project_to_tangent=True):
    """Backward path residual for overdamped dynamics."""
    mu_b = x_next + drift * dt / 2

    G_list = project_tangent(manifold, x_prev - mu_b, x_next, project_to_tangent=project_to_tangent)
    G_list = G_list / (sigmas * math.sqrt(dt))

    return G_list

def G_bwd_underdamped(manifold, x_prev, x_next, v_next, sigmas, dt, drift, a, sqrt_1ma2):
    mu_b = a * v_next + (sigmas ** 2) * dt * drift

    mu_b = x_next - (sigmas ** 2) * dt * mu_b

    G_list = project_tangent(manifold, x_prev - mu_b, x_next)
    G_list = G_list / (sqrt_1ma2 * (sigmas ** 2) * dt)
    
    return G_list


def loss_overdamped_path(manifold, x_hist, score_net, func_b, sigmas, dt, project_to_tangent=True):
    """Path matching loss for overdamped trajectories."""
    N = sigmas.shape[0]
    B = x_hist.shape[1]
    device = x_hist.device
    sigmas = sigmas.repeat_interleave(B).reshape(-1, 1)
    
    t_vec = torch.linspace(0., N*dt, N+1, device=device)[1:]
    t_full = t_vec[:, None].expand(N, B).reshape(-1)

    x_next = x_hist[1:].reshape(-1, x_hist.size(-1))
    x_prev = x_hist[:-1].reshape_as(x_next)

    b_val = func_b(x_next)
    scores = score_net(x_next, t_full)

    drift = sigmas**2 * (scores - b_val)

    G_bwd_list = G_bwd_overdamped(manifold, x_prev, x_next, sigmas, dt, drift, project_to_tangent=project_to_tangent)
    G_bwd_sq = (G_bwd_list**2).sum(-1)
    return 0.5 * G_bwd_sq.view(N, B).sum(0).mean()

def loss_underdamped_path(manifold, x_hist, score_net, func_b, sigmas, dt, gamma):
    """Path matching loss for underdamped trajectories."""
    N = sigmas.shape[0]
    B, D = x_hist.shape[1], x_hist.shape[2]
    device = x_hist.device

    t_vec = torch.linspace(0., N*dt, N+1, device=device)[1:]
    t_full = t_vec[:, None].expand(N, B).reshape(-1)

    x_prev = x_hist[:-1]
    x_next = x_hist[1:]

    v_bwd = (x_hist[2:] - x_hist[1:-1]) / ((sigmas[1:, None, None] ** 2) * dt)

    vN = torch.randn_like(x_hist[-1], device=device)
    v_bwd = torch.cat([v_bwd, vN.unsqueeze(0)], dim=0)

    x_prev = x_prev.reshape(-1, D)
    x_next = x_next.reshape(-1, D)
    v_bwd = v_bwd.reshape(-1, D)
    v_bwd = project_tangent(manifold, v_bwd, base=x_next)

    sigmas_step = sigmas.repeat_interleave(B).reshape(-1, 1)

    a = torch.exp(- (sigmas_step ** 2) * gamma * dt)
    sqrt_1ma2 = torch.sqrt(torch.abs(1.0 - a**2))

    b_val = func_b(x_next)
    scores = score_net(torch.cat([x_next, v_bwd], dim=-1), t_full)
    drift = scores - b_val

    G_bwd_list = G_bwd_underdamped(manifold, x_prev, x_next, v_bwd, sigmas_step, dt, drift, a, sqrt_1ma2)
    G_bwd_sq = (G_bwd_list**2).sum(-1)
    return 0.5 * G_bwd_sq.view(N, B).sum(0).mean()
