import torch
from torch.func import vmap, jacrev

batched_jacobian = lambda f: vmap(jacrev(f))

class Manifold_general:
    """
    A general manifold with equality h(x)=0 and inequality g(x)<=0 constraints.
    This class uses an internal cache and fully vectorized operations to maximize performance.
    """
    def __init__(self, dim, m, l, h, g, grad_h=None, grad_g=None, boundary_repulsion_rate=0.1):
        self.dim = dim
        self.m = m
        self.l = l
        self.out_dim = dim
        
        self.h_single = h
        self.g_single = g

        self.h = vmap(h) if m > 0 else None
        self.g = vmap(g) if l > 0 else None

        self.cut_off = 0.00
        self.epsilon = boundary_repulsion_rate

        self.grad_h = grad_h if grad_h is not None else (batched_jacobian(self.h_single) if m > 0 else lambda x: torch.empty(x.shape[0], 0, self.dim, device=x.device, dtype=x.dtype))
        self.grad_g = grad_g if grad_g is not None else (batched_jacobian(self.g_single) if l > 0 else lambda x: torch.empty(x.shape[0], 0, self.dim, device=x.device, dtype=x.dtype))

        self._cached_x = None
        self._cached_nabla_J = None
        self._cached_gram_matrix = None
        self._cached_active_constraints_mask = None
        self._cached_J_values = None

    def _update_geometry_cache(self, x):
        """Internal method to compute and cache geometric quantities for a point x."""
        if (self._cached_x is not None and self._cached_x.shape == x.shape and torch.allclose(self._cached_x, x, atol=1e-7)):
            return

        bsz, dim = x.shape
        total_constraints = self.m + self.l

        nabla_J = torch.zeros(bsz, total_constraints, dim, device=x.device, dtype=x.dtype)
        J_values = torch.zeros(bsz, total_constraints, device=x.device, dtype=x.dtype)
        active_mask = torch.zeros(bsz, total_constraints, dtype=torch.bool, device=x.device)

        if self.m > 0:
            nabla_J[:, :self.m, :] = self.grad_h(x)
            J_values[:, :self.m] = self.h(x)
            active_mask[:, :self.m] = True

        if self.l > 0:
            g_vals = self.g(x)
            active_g_mask = g_vals >= -self.cut_off
            nabla_J[:, self.m:, :] = self.grad_g(x)
            J_values[:, self.m:] = g_vals
            active_mask[:, self.m:] = active_g_mask
        
        nabla_J[~active_mask] = 0.0
        
        nabla_J_T = nabla_J.transpose(1, 2)
        gram_matrix = torch.bmm(nabla_J, nabla_J_T)
        gram_matrix.diagonal(dim1=-2, dim2=-1).add_(1e-6)

        self._cached_x = x
        self._cached_nabla_J = nabla_J
        self._cached_gram_matrix = gram_matrix
        self._cached_active_constraints_mask = active_mask
        self._cached_J_values = J_values
    
    def project_onto_tangent_space(self, y, base_point):
        """Projects a vector v onto the tangent space at x, using cached geometry."""
        self._update_geometry_cache(base_point)
        nabla_J, gram_matrix, active_mask = self._cached_nabla_J, self._cached_gram_matrix, self._cached_active_constraints_mask
        if not active_mask.any():
             return y

        nabla_J_v = torch.bmm(nabla_J, y.unsqueeze(-1))
        tang_vec = torch.linalg.lstsq(gram_matrix, nabla_J_v).solution
        return y - torch.bmm(nabla_J.transpose(1, 2), tang_vec).squeeze(-1)

    def constraint_fn(self, samples):
        """Computes the value of the equality constraint function h(x)."""
        if self.m > 0:
            return self.h(samples)
        else:
            return torch.empty(samples.shape[0], 0, device=samples.device, dtype=samples.dtype)

    def adding_correction_decaying(self, y, base_point, delta_t, alpha, sigma_sq):
        """Adds a correction term to y, using cached geometry."""
        self._update_geometry_cache(base_point)
        nabla_J, gram_matrix, J_values, active_mask = self._cached_nabla_J, self._cached_gram_matrix, self._cached_J_values, self._cached_active_constraints_mask
        
        if not active_mask.any():
            return base_point + y
            
        J_values_decay = J_values.clone()
        if self.l > 0:
            J_values_decay[:, self.m:] += self.epsilon

        masked_J_decay = torch.where(active_mask, J_values_decay, torch.zeros_like(J_values_decay))
        z = torch.linalg.lstsq(gram_matrix, masked_J_decay.unsqueeze(-1)).solution
        decaying_term = -alpha * torch.bmm(nabla_J.transpose(1, 2), z).squeeze(-1)

        scaling_factor = sigma_sq * torch.abs(delta_t)
        if scaling_factor.ndim == 1: scaling_factor = scaling_factor.unsqueeze(1)
            
        return base_point + y + decaying_term * scaling_factor
    
    def adding_correction_decaying_implicit(self, base_point, delta_t, alpha, sigma_sq):
        """Adds a correction term to y, using cached geometry."""
        # Same scale as explicit correction with alpha = 1 / (sigma_sq * |dt|) at this point.
        self._update_geometry_cache(base_point)
        nabla_J, gram_matrix, J_values, active_mask = self._cached_nabla_J, self._cached_gram_matrix, self._cached_J_values, self._cached_active_constraints_mask
        
        if not active_mask.any():
            return base_point 
            
        J_values_decay = J_values.clone()
        if self.l > 0:
            J_values_decay[:, self.m:] += self.epsilon

        masked_J_decay = torch.where(active_mask, J_values_decay, torch.zeros_like(J_values_decay))
        z = torch.linalg.lstsq(gram_matrix, masked_J_decay.unsqueeze(-1)).solution

        decaying_term = - torch.bmm(nabla_J.transpose(1, 2), z).squeeze(-1)
        return base_point + decaying_term

    @torch.no_grad()
    def project_onto_manifold_with_base(self, y, base_point, threshold=1e-5, n_iters=30, **kwargs):
        """Projects a point y + base_point onto the manifold using Newton's method."""
        x_proj = y + base_point
        tol = min(float(threshold), 1e-6)

        I_prev = torch.zeros(x_proj.size(0), self.l, dtype=torch.bool, device=x_proj.device)
        for i in range(n_iters):
            self._update_geometry_cache(x_proj)
            nabla_J, gram, J_vals, active_mask = (
                self._cached_nabla_J, self._cached_gram_matrix,
                self._cached_J_values, self._cached_active_constraints_mask
            )

            h_vals = J_vals[:, :self.m]
            g_vals = J_vals[:, self.m:]

            near_active = (g_vals >= -self.cut_off)
            violated = (g_vals > 0)
            I = near_active | violated | I_prev

            mask = torch.zeros_like(J_vals, dtype=torch.bool)
            mask[:, :self.m] = True
            mask[:, self.m:] = I

            nabla_J_eff = nabla_J.clone()
            nabla_J_eff[~mask] = 0.0
            gram_eff = torch.bmm(nabla_J_eff, nabla_J_eff.transpose(1,2))
            gram_eff.diagonal(dim1=-2, dim2=-1).add_(1e-6)

            rhs = torch.where(mask, J_vals, torch.zeros_like(J_vals)).unsqueeze(-1)
            lambda_mu = torch.linalg.lstsq(gram_eff, rhs).solution

            correction = -torch.bmm(nabla_J_eff.transpose(1,2), lambda_mu).squeeze(-1)
            x_new = x_proj + correction

            self._update_geometry_cache(x_new)
            nabla_J_eff = self._cached_nabla_J.clone()
            nabla_J_eff[~mask] = 0.0
            gram_eff = torch.bmm(nabla_J_eff, nabla_J_eff.transpose(1,2))
            gram_eff.diagonal(dim1=-2, dim2=-1).add_(1e-6)
            rhs = torch.where(mask, self._cached_J_values, torch.zeros_like(J_vals)).unsqueeze(-1)
            lambda_mu = torch.linalg.lstsq(gram_eff, rhs).solution

            mu_block = lambda_mu[:, self.m:, 0]
            drop = (mu_block < -1e-8)
            I[drop] = False
            I_prev = I.clone()

            x_proj = x_new

            eq_feas = (h_vals.abs().max(dim=1).values < tol)
            ineq_feas = (g_vals.max(dim=1).values <= tol)
            comp_res = torch.maximum(g_vals, torch.zeros_like(g_vals)) * torch.clamp(mu_block, min=0).max(dim=1).values
            done = eq_feas & ineq_feas & (comp_res < tol)
            if torch.all(done):
                break

        self._update_geometry_cache(x_proj)
        J_values_final, active_mask_final = self._cached_J_values, self._cached_active_constraints_mask

        final_violations = torch.abs(J_values_final)
        final_violations[~active_mask_final] = 0.0

        non_converged_flag = torch.any(final_violations > threshold, dim=1) | ~torch.all(torch.isfinite(x_proj), dim=1)

        x_proj[non_converged_flag] = base_point[non_converged_flag]
        return x_proj.detach(), torch.logical_not(non_converged_flag).to(y)
