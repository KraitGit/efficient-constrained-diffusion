import logging
import torch


@torch.no_grad()
def refine_dataset_SDF(sdf, samples0, tol=1e-5, max_iter_n=1000, step_size=1e-1, keep_quiet=False):
    @torch.enable_grad()
    def sdf_grad(samples):
        samples.requires_grad_(True)
        gradients = torch.autograd.grad(
            outputs=sdf(samples).sum(),
            inputs=samples,
            create_graph=True,
            retain_graph=True,
        )[0]
        return gradients.detach()

    if isinstance(samples0, torch.Tensor):
        samples = samples0.clone()
        device = samples.device
    else:
        samples = torch.tensor(samples0, dtype=torch.float32)
        device = torch.device("cpu")

    active_idx = torch.arange(samples.shape[0], dtype=torch.int64, device=device)

    iter_n = 0
    for iter_n in range(max_iter_n):
        xi_vals = sdf(samples[active_idx])
        error = torch.abs(xi_vals).squeeze(dim=1)
        bad_idx = error >= tol
        if bad_idx.sum() == 0:
            break
        if iter_n % 50 == 0 and not keep_quiet:
            logging.info(f"iter {iter_n}: max_err={torch.max(error):.3e}, {bad_idx.sum()} bad states, tol={tol:.3e}")
        active_idx = active_idx[bad_idx]
        samples[active_idx] = samples[active_idx] - xi_vals[bad_idx] * sdf_grad(samples[active_idx]) * step_size

    max_error = torch.max(torch.abs(sdf(samples)).squeeze())
    if not keep_quiet:
        logging.info(f"Total steps={iter_n}, final error: {max_error:.3e}.")
        if max_error > tol * 1.1:
            logging.warning(f"Tolerance ({tol:.3e}) not reached.")

    return samples.detach().cpu()
