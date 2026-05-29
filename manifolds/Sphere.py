import torch
import numpy as np
import logging


def latlon_to_xyz(data):
    """
    :data[:, 0]: [-90, 90], latitude
    :data[:, 1]: [-180, 180], longitude
    
    :return: 3D point in S2
    :theta: [0, pi]
    :phi: [-pi, pi]
    """
    theta = (90 - data[:, 0]) * np.pi / 180
    phi = (data[:, 1]) * np.pi / 180

    if isinstance(theta, torch.Tensor):
        lib = torch
        concatenate = torch.cat
    else:
        lib = np
        concatenate = np.concatenate

    x = lib.sin(theta) * lib.cos(phi)
    y = lib.sin(theta) * lib.sin(phi)
    z = lib.cos(theta)
    return concatenate([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], 1)


def xyz_to_latlon(points):
    """
    points: 3D point in S2
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    if isinstance(points, torch.Tensor):
        acos = torch.acos
        atan2 = torch.atan2
    else:
        acos = np.arccos
        atan2 = np.arctan2
    
    if isinstance(points, torch.Tensor):
        if torch.isnan(z).any():
            logging.warning("NaN values detected in z coordinates")
    else:
        if np.isnan(z).any():
            logging.warning("NaN values detected in z coordinates")

    theta = acos(z)
    phi = atan2(y, x)
    lat, lon = 90 - theta * 180 / np.pi, phi * 180 / np.pi
    return lat, lon


class Manifold_Sphere:
    def __init__(self, dim):
        self.out_dim = dim + 1
        self.inner_dim = dim

    def constraint_fn(self, samples):
        return samples.norm(dim=1, keepdim=True) - 1

    def constraint_grad_fn(self, samples):
        return samples/samples.norm(dim=1, keepdim=True)

    def project_onto_tangent_space(self, y, base_point, **kwargs):
        coeff = torch.sum(y * base_point, dim=1, keepdim=True) / (base_point**2).sum(dim=1, keepdim=True)
        return y - coeff * base_point

    def adding_correction_decaying(self, y, base_point, delta_t, alpha, sigma_sq):
        h_val = self.constraint_fn(base_point)
        h_grad = self.constraint_grad_fn(base_point)
        grad_h_norm_sq = (h_grad**2).sum(dim=1, keepdim=True)
        decaying_term = - alpha * h_grad * h_val.reshape(-1, 1) / grad_h_norm_sq
        return base_point + y + decaying_term * sigma_sq * torch.abs(delta_t)

    def adding_correction_decaying_implicit(self, base_point, delta_t, alpha, sigma_sq):
        # Same scale as explicit correction with alpha = 1 / (sigma_sq * |dt|) at this point.
        h_val = self.constraint_fn(base_point)
        h_grad = self.constraint_grad_fn(base_point)
        grad_h_norm_sq = (h_grad**2).sum(dim=1, keepdim=True)
        decaying_term = - h_grad * h_val.reshape(-1, 1) / grad_h_norm_sq
        return base_point + decaying_term


    def project_onto_manifold_with_base(self, y, base_point, **kwargs):
        """
        Proj(x+v/(1-|v|^2)^(1/2))
        """
        if (y.norm(dim=1) > 1).any():
            bad_idx = torch.where(y.norm(dim=1) > 1)[0]
            logging.info(f'Warning: index {bad_idx.detach().cpu()} of v can not be projected! The max norm of v: {y.norm(dim=1).max():.4f}.')
            converged_flag =(y.norm(dim=1) < 1)
            y[bad_idx, :] = y[bad_idx, :] * 0.99 / y[bad_idx, :].norm(dim=1).max()
        else:
            converged_flag = torch.ones(y.shape[0], dtype=torch.bool)

        temp = base_point + y/torch.sqrt(1-(y**2).sum(dim=1, keepdim=True))
        return temp / temp.norm(dim=1, keepdim=True), converged_flag.to(y)

    def uniform_sample(self, sample_num):
        point = torch.randn((sample_num, self.out_dim))
        return point / (point.norm(dim=1, keepdim=True) + 1e-6)
