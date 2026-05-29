import torch
import numpy as np
import trimesh
import logging
from src.utils import uniform_triangles_sample


class Manifold_SDF:
    def __init__(self, model_path, mesh_path):
        self.out_dim = 3
        self.inner_dim = 2
        self.model = torch.load(model_path, map_location='cpu', weights_only=False)
        self.mesh = trimesh.load(mesh_path)
        self.area = self.mesh.area_faces.sum()
        self.probs = self.mesh.area_faces / self.area

    def constraint_fn(self, samples):
        return self.model(samples)

    @torch.enable_grad()
    def constraint_grad_fn(self, samples):
        samples.requires_grad_(True)
        gradients = torch.autograd.grad(
            outputs=self.constraint_fn(samples).sum(),
            inputs=samples,
            create_graph=True,
            retain_graph=True)[0]
        return gradients.detach()

    def adding_correction_decaying(self, y, base_point, delta_t, alpha, sigma_sq):
        h_val = self.constraint_fn(base_point)
        h_grad = self.constraint_grad_fn(base_point)
        norm_sq_h_grad = (h_grad**2).sum(dim=1, keepdim=True)
        decaying_term = -alpha * h_grad * h_val.reshape(-1, 1) / norm_sq_h_grad
        return base_point + y + decaying_term * torch.abs(delta_t) * sigma_sq

    def adding_correction_decaying_implicit(self, base_point, delta_t, alpha, sigma_sq):
        # Same scale as explicit correction with alpha = 1 / (sigma_sq * |dt|) at this point.
        h_val = self.constraint_fn(base_point)
        h_grad = self.constraint_grad_fn(base_point)
        grad_h_norm_sq = (h_grad**2).sum(dim=1, keepdim=True)
        decaying_term = -h_grad * h_val.reshape(-1, 1) / grad_h_norm_sq
        return base_point + decaying_term


    def project_onto_tangent_space(self, y, base_point):
        """
        The grad norm is close to 1!
        """
        norm_vec = self.constraint_grad_fn(base_point)
        coeff = torch.sum(y * norm_vec, dim=1, keepdim=True) / (norm_vec**2).sum(dim=1, keepdim=True)
        return y - coeff * norm_vec

    @torch.no_grad()
    def project_onto_manifold_with_base(self, y, base_point, threshold=1e-4, n_iters=10, **kwargs):
        """
            Here, y is a tangent vector.
            Find mu such that xi(y + base_point + mu grad xi(base_point)) = 0.
        """
        keep_quiet = kwargs["keep_quiet"] if "keep_quiet" in kwargs else True

        grad_vec = self.constraint_grad_fn(base_point)
        mu = torch.zeros(y.shape[0], 1).to(y)
        active_idx = torch.arange(0, y.shape[0], dtype=torch.int64).to(y.device)

        for i in range(n_iters):
            temp = y[active_idx,:] + base_point[active_idx,:] - grad_vec[active_idx,:] * mu[active_idx,:]
            value = self.constraint_fn(temp)
            bad_idx = (value.abs() >= threshold).squeeze(dim=1)
            if bad_idx.sum() == 0 and (i > 1):
                break
            active_idx = active_idx[bad_idx]
            mu_grad = - (self.constraint_grad_fn(temp[bad_idx,:]) * grad_vec[active_idx,:]).sum(dim=1, keepdim=True)
            mu[active_idx,:] = mu[active_idx,:] - value[bad_idx, :] / mu_grad

        projected_pt = y + base_point - grad_vec * mu
        value = self.constraint_fn(projected_pt).abs().squeeze()

        non_converged_flag = value > threshold
        non_converged_num = non_converged_flag.sum() 

        projected_pt[non_converged_flag] = base_point[non_converged_flag]

        if not keep_quiet:
            logging.info(f'total steps: {i}, max_error: {value.max():.3e}, {non_converged_num} states not converged!')
        return projected_pt.detach(), torch.logical_not(non_converged_flag).to(y)

    def uniform_sample(self, sample_num):
        inds = np.random.choice(int(self.probs.shape[0]), int(sample_num), p=self.probs)
        triangles = self.mesh.triangles[inds]
        samples = uniform_triangles_sample(triangles)
        return torch.tensor(samples, dtype=torch.float32)
