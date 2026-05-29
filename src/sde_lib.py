import torch


class SDE_Brownian_manifolds:
    def __init__(self, sigma_min, sigma_max, N, T, sampler=None, drift_mode='zero', sigma_schedule='geometric'):

        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N = N
        self.T = T
        self.dt = self.T/self.N
        self.sigma_schedule = sigma_schedule.lower()
        if drift_mode == 'zero':
            self.func_b = lambda x: 0 * x
        elif drift_mode == 'linear':
            self.func_b = lambda x: -x

        self.sampler = sampler

        self.g_0 = self.sde(None, torch.tensor([0.]))[1]
        self.g_T = self.sde(None, torch.tensor([self.T]))[1]

    def get_diffusion(self, t):
        return self.get_sigma(t)

    def get_sigma(self, t):
        """
        Evaluate sigma according to the selected schedule.
        Supported schedules:
            'linear'    : linear interpolation between sigma_min and sigma_max.
            'geometric' : power-law interpolation sigma_min * (sigma_max/sigma_min)^tau.
        """
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, dtype=torch.float32)

        tau = torch.clamp(t / self.T, 0.0, 1.0)
        sigma_min = torch.as_tensor(self.sigma_min, device=tau.device, dtype=tau.dtype)
        sigma_max = torch.as_tensor(self.sigma_max, device=tau.device, dtype=tau.dtype)

        if self.sigma_schedule in ('linear', 'lin'):
            sigma = sigma_min + tau * (sigma_max - sigma_min)
        elif self.sigma_schedule in ('geometric', 'power', 'pow'):
            ratio = sigma_max / sigma_min
            sigma = sigma_min * torch.pow(ratio, tau)
        else:
            raise ValueError(f"Unknown sigma schedule '{self.sigma_schedule}'.")
        return sigma
    
    def drift_b(self, x):
        return self.func_b(x).to(x)
    
    def sde(self, x, diff_t):
        diffusion = self.get_diffusion(diff_t)
        
        if x is None:
            drift = 0.
        else:
            temp = self.func_b(x).to(x)
            if len(temp.shape) == 2:
                drift = diffusion.reshape(-1, 1)**2 * temp
            else:
                drift = diffusion.reshape(-1, 1, 1) ** 2 * temp

        return drift, diffusion if x is None else diffusion.to(x)

    def reverse(self, score_fn, underdamped=False):
        N = self.N
        T = self.T
        sde_fn = self.sde

        if underdamped:
            class RSDE(self.__class__):
                def __init__(self):
                    self.N = N
                    self.T = T

                def drift_score(self, x, v, score_t):
                    score = score_fn(torch.cat([x, v], dim = -1), label = score_t)
                    return score

                def sde(self, x, v, score_t, diff_t):
                    drift, diffusion = sde_fn(x, diff_t)

                    score = score_fn(torch.cat([x, v], dim=-1), label = score_t)

                    if len(score.shape) == 2:
                        drift = drift - diffusion[:, None] ** 2 *   score
                    else:
                        drift = drift -  diffusion[:, None, None] ** 2 * score

                    return drift, diffusion
        else:
            class RSDE(self.__class__):
                def __init__(self):
                    self.N = N
                    self.T = T

                def sde(self, x, score_t, diff_t):
                    """Create the drift and diffusion functions for the reverse SDE."""
                    drift, diffusion = sde_fn(x, diff_t)
                    score = score_fn(x, score_t) 

                    if len(score.shape) == 2:
                        drift = drift - diffusion[:, None] ** 2 *   score
                    else:
                        drift = drift -  diffusion[:, None, None] ** 2 * score

                    return drift, diffusion

        return RSDE()
