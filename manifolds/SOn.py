import torch
import numpy as np
import logging

DAMPING_CONST = 0

class Manifold_SOn:
    def __init__(self, dim):
        self.mat_dim = dim
        self.out_dim = self.mat_dim * self.mat_dim
        self.inner_dim = int(self.mat_dim * (self.mat_dim - 1) / 2)

        self.triu_indices = torch.triu_indices(self.mat_dim, self.mat_dim)
        self.constraint_num = self.triu_indices.shape[1]
        self.grad_coeff_matrix_normalized = self.get_grad_coeff_matrix(normalized=True)
        self.grad_coeff_matrix = self.get_grad_coeff_matrix(normalized=False)

    def get_grad_coeff_matrix(self, normalized):
        mat = torch.zeros(self.constraint_num, self.mat_dim, self.mat_dim)

        alpha = np.sqrt(0.5)
        for idx in range(self.constraint_num):
            i, j = self.triu_indices[:, idx]
            if normalized:
                if i == j:
                    mat[idx, i, j] += 1.
                else:
                    mat[idx, i, j] += alpha
                    mat[idx, j, i] += alpha
            else:
                mat[idx, i, j] += 1.0
                mat[idx, j, i] += 1.0
        return mat

    def constraint_fn(self, samples):
        samples = samples.reshape(-1, self.mat_dim, self.mat_dim)
        temp = torch.bmm(samples, torch.transpose(samples, dim0=1, dim1=2)) - torch.eye(self.mat_dim, self.mat_dim).to(samples)
        return temp[:, self.triu_indices[0, :], self.triu_indices[1, :]]

    def constraint_grad_fn(self, samples, normalized=False):
        samples = samples.reshape(-1, self.mat_dim, self.mat_dim).unsqueeze(1)
        if normalized:
            temp = torch.matmul(self.grad_coeff_matrix_normalized.to(samples), samples).flatten(
                start_dim=-2)
        else:
            temp = torch.matmul(self.grad_coeff_matrix.to(samples), samples).flatten(
                start_dim=-2)
        return temp

    def project_onto_tangent_space(self, y, base_point):
        """
        P_X(U) = (U-XU^TX)/2
        """
        y = y.reshape(-1, self.mat_dim, self.mat_dim)
        base_point = base_point.reshape(-1, self.mat_dim, self.mat_dim)
        out = (y - torch.bmm(base_point, torch.bmm(torch.transpose(y, dim0=1, dim1=2), base_point))) * 0.5
        return out.reshape(-1, self.out_dim)

    def adding_correction_decaying(self, y, base_point, delta_t, alpha, sigma_sq):
        h_val = self.constraint_fn(base_point)
        h_grad = self.constraint_grad_fn(base_point)
        G = torch.bmm(h_grad, h_grad.transpose(1, 2)) + DAMPING_CONST * torch.eye(h_grad.shape[1]).to(h_grad.device)
        coeff = torch.linalg.lstsq(G, h_val.unsqueeze(-1)).solution
        decaying_term = -alpha * torch.bmm(h_grad.transpose(1, 2), coeff).squeeze()
        return base_point + y + decaying_term * sigma_sq * torch.abs(delta_t)

    def adding_correction_decaying_implicit(self, base_point, delta_t, alpha, sigma_sq):
        # Same scale as explicit correction with alpha = 1 / (sigma_sq * |dt|) at this point.
        h_val = self.constraint_fn(base_point)
        h_grad = self.constraint_grad_fn(base_point)
        G = torch.bmm(h_grad, h_grad.transpose(1, 2)) + DAMPING_CONST * torch.eye(h_grad.shape[1]).to(h_grad.device)
        coeff = torch.linalg.lstsq(G, h_val.unsqueeze(-1)).solution
        decaying_term = -torch.bmm(h_grad.transpose(1, 2), coeff).squeeze()
        return base_point + decaying_term

    @torch.no_grad()
    def project_onto_manifold_with_base(self, y, base_point, threshold=1e-6, n_iters=10, **kwargs):

        keep_quiet = kwargs.get("keep_quiet", True)
        
        grad_vec_full = self.constraint_grad_fn(base_point, normalized=False)
        mu = torch.zeros((y.shape[0], self.constraint_num)).to(y)
        active_idx = torch.arange(y.shape[0], dtype=torch.int64, device=y.device)

        for i in range(n_iters):
            if active_idx.shape[0] == 0:
                break
            
            active_y = y[active_idx, :]
            active_base_point = base_point[active_idx, :]
            active_grad_vec = grad_vec_full[active_idx, :]
            active_mu = mu[active_idx, :]
            
            temp = active_y + active_base_point - torch.einsum('ijk,ij->ik', active_grad_vec, active_mu)
            value = self.constraint_fn(temp)
            
            bad_mask = value.norm(dim=1) >= threshold
            
            if not torch.any(bad_mask):
                break
                
            bad_indices_in_active = torch.where(bad_mask)[0]
            
            grad_vec_bad = active_grad_vec[bad_indices_in_active, :]
            temp_bad = temp[bad_indices_in_active, :]
            value_bad = value[bad_indices_in_active, :]
            
            mu_grad = -torch.bmm(self.constraint_grad_fn(temp_bad, normalized=False),
                                  grad_vec_bad.transpose(1, 2))
            delta_mu = torch.linalg.lstsq(mu_grad, value_bad).solution
            current_mu_bad = active_mu[bad_indices_in_active, :]
            mu[active_idx[bad_indices_in_active], :] = current_mu_bad - delta_mu

            active_idx = active_idx[bad_indices_in_active]

        projected_pt = y + base_point - torch.einsum('ijk,ij->ik', grad_vec_full, mu)
        value = self.constraint_fn(projected_pt).abs()

        non_converged_flag = torch.any((value > threshold) | (~torch.isfinite(value)), dim=1)
        non_converged_num = non_converged_flag.sum()

        projected_pt[non_converged_flag, :] = base_point[non_converged_flag, :]

        if not keep_quiet:
            logging.info(f'total steps: {i+1}, max_error: {value.max().item():.3e}, {non_converged_num} states not converged!')
            
        return projected_pt.detach(), torch.logical_not(non_converged_flag).to(y)

    def uniform_sample(self, sample_num):
        """
        Ensure the matrices are in the correct component
        """
        sample = torch.tensor([])
        while sample.shape[0] < sample_num:
            Z = torch.randn(sample_num, self.mat_dim, self.mat_dim)
            idx1 = torch.where(torch.linalg.det(Z).abs() > 1e-4)[0]
            Q, R = torch.linalg.qr(Z[idx1], mode="complete")
            diag = torch.diag_embed(R.diagonal(dim1=-2, dim2=-1).sign())
            Q = torch.bmm(Q, diag)
            idx2 = torch.where(torch.linalg.det(Q) > 0)[0]
            sample = torch.cat((sample, Q[idx2]), dim=0)
        return sample[:sample_num].reshape(-1, self.out_dim)
