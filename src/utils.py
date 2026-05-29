import logging
import math
import os
import shutil

import numpy as np
import torch
import torch.distributed as dist
from scipy.spatial.distance import jensenshannon
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


### Logging ###

def get_logger(log_dir, verbose='info', rank=0):
    if rank == 0 and os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    level = getattr(logging, verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(verbose))

    log_name = 'log.txt' if rank == 0 else f'log_rank{rank}.txt'
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s',
        datefmt='%m-%d %H:%M:%S',
    )

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_dir, log_name))
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(level)
    for name in ["kaleido", "choreographer", "logistro", "chrome_wrapper"]:
        logging.getLogger(name).setLevel(logging.WARNING)
    return logger


### Distributed / Device ###

def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return False, 0, 1, None

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def select_device(config_gpu, distributed=False, local_rank=None):
    if not torch.cuda.is_available():
        return "cpu", None
    if distributed:
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}", local_rank
    if config_gpu is not None:
        torch.cuda.set_device(config_gpu)
        return f"cuda:{config_gpu}", config_gpu
    return "cuda", None


def synchronize_device(device):
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def distributed_barrier(distributed):
    if distributed and dist.is_initialized():
        dist.barrier()


def maybe_wrap_ddp(model, distributed, local_rank):
    if not distributed:
        return model
    device_ids = [local_rank] if torch.cuda.is_available() else None
    return DDP(model, device_ids=device_ids)


def reduce_mean_float(value, device, distributed=False, world_size=1):
    if not distributed:
        return float(value)
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / world_size).item()


### Models ###

def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def model_parameters(model):
    return unwrap_model(model).parameters()


def save_model(workdir, network, name):
    save_path = os.path.join(workdir, name)
    torch.save(unwrap_model(network), save_path)


def load_model(model_path):
    if not os.path.exists(model_path):
        logging.info(f'The trained model path {model_path} does not exist.')
        return None

    logging.info(f"Loading model from {model_path}")
    return torch.load(
        model_path,
        map_location=torch.device('cpu'),
        weights_only=False,
    )


class ExponentialMovingAverage:
    def __init__(self, parameters, decay, use_num_updates=True):
        if not 0.0 <= decay <= 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach() for p in parameters]
        self.collected_params = []

    def update(self, parameters):
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)


### Data ###

def split_dataset(data_ori, data_seed):
    total_size = data_ori.shape[0]
    test_size = int(total_size * 0.1)
    val_size = int(total_size * 0.1)
    training_size = total_size - test_size - val_size

    train_data, val_data, test_data = torch.utils.data.random_split(
        data_ori,
        [training_size, val_size, test_size],
        generator=torch.Generator().manual_seed(data_seed),
    )

    training_set = data_ori[train_data.indices]
    test_set = data_ori[test_data.indices]
    val_set = data_ori[val_data.indices]
    logging.info(
        f"size of training set: {training_set.shape[0]}, "
        f"size of validation set: {val_set.shape[0]}, "
        f"size of test set: {test_set.shape[0]}."
    )
    return training_set, test_set, val_set


def make_index_loader(num_items, batch_size, distributed=False, world_size=1, rank=0, already_sharded=False):
    dataset = torch.arange(num_items)
    if distributed:
        batch_size = max(1, math.ceil(batch_size / world_size))
    sampler = None
    if distributed and not already_sharded:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
    )
    return loader, sampler


def sample_prior(num_samples, sampler_fn, device):
    samples = sampler_fn(num_samples)
    return samples.to(device) if samples.device != device else samples


def format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0


def check_memory(data=None, keep_quiet=False):
    if data is not None:
        memory_bytes = data.element_size() * data.nelement()
        if not keep_quiet:
            logging.info(f"The data (shape {data.shape}) occupy {format_bytes(memory_bytes)} of memory on {data.device}.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if not keep_quiet:
            logging.info(f"CUDA memory allocated: {format_bytes(torch.cuda.memory_allocated())}")


### Metrics / Reporting ###

def get_constraint_metrics(manifold, samples):
    out_dim = getattr(manifold, 'out_dim', samples.shape[-1])
    positions = samples[..., :out_dim]

    eq_vals = manifold.h(positions) if getattr(manifold, 'h', None) is not None else None
    if (eq_vals is None or eq_vals.numel() == 0) and hasattr(manifold, 'constraint_fn'):
        eq_vals = manifold.constraint_fn(positions)

    ineq_vals = manifold.g(positions) if getattr(manifold, 'g', None) is not None else None

    metrics = {}
    if isinstance(eq_vals, torch.Tensor) and eq_vals.numel() > 0:
        metrics["mean_eq"] = eq_vals.abs().mean().item()
    if isinstance(ineq_vals, torch.Tensor) and ineq_vals.numel() > 0:
        metrics["mean_ineq"] = torch.relu(ineq_vals).mean().item()
    return metrics


def log_constraint_metrics(manifold, samples, prefix="backward"):
    metrics = get_constraint_metrics(manifold, samples)
    if "mean_eq" in metrics:
        logging.info(f"{prefix} mean_eq_viol.: {metrics['mean_eq']:.4e}")
    if "mean_ineq" in metrics:
        logging.info(f"{prefix} mean_ineq_viol.: {metrics['mean_ineq']:.4e}")


def get_runner_timing(runner):
    training_per_sample = runner.training_time_total / runner.training_sample_count if runner.training_sample_count else 0.0
    sampling_per_sample = runner.trajectory_gen_time / runner.trajectory_sample_count if runner.trajectory_sample_count else 0.0
    return {
        "training_total": runner.training_time_total,
        "training_per_sample": training_per_sample,
        "sampling_total": runner.trajectory_gen_time,
        "sampling_per_sample": sampling_per_sample,
    }


def log_runner_sampling_timing(runner, prefix):
    timing = get_runner_timing(runner)
    logging.info(
        f"{prefix} sampling_time: total={timing['sampling_total']:.2f}s, "
        f"per_sample={timing['sampling_per_sample']:.6f}s/sample, samples={runner.trajectory_sample_count}"
    )


def log_task_metric(prefix, name, value):
    logging.info(f"{prefix} {name}: {float(value):.6f}")


def log_validation_summary(prefix, runner, constraints=None, metric_groups=None):
    timing = get_runner_timing(runner)
    parts = [
        prefix,
        f"train_time={timing['training_total']:.2f}s ({timing['training_per_sample']:.2e}s/sample)",
        f"sampling_time={timing['sampling_total']:.2f}s ({timing['sampling_per_sample']:.2e}s/sample)",
    ]

    constraints = constraints or {}
    if "mean_eq" in constraints:
        parts.append(f"mean_eq={constraints['mean_eq']:.2e}")
    if "mean_ineq" in constraints:
        parts.append(f"mean_ineq={constraints['mean_ineq']:.2e}")

    for name, value in metric_groups or []:
        parts.append(f"{name}={float(value):.6f}")

    logging.info(" | ".join(parts))


def compute_jsd_2d_histogram(xs, xt, bins=30, ranges=None):
    p_hist, _, _ = np.histogram2d(xs[:, 0], xs[:, 1], bins=bins, range=ranges)
    q_hist, _, _ = np.histogram2d(xt[:, 0], xt[:, 1], bins=bins, range=ranges)

    p_dist = p_hist.ravel()
    q_dist = q_hist.ravel()
    p_dist = p_dist / p_dist.sum()
    q_dist = q_dist / q_dist.sum()

    epsilon = 1e-10
    p_dist = p_dist + epsilon
    q_dist = q_dist + epsilon
    p_dist = p_dist / p_dist.sum()
    q_dist = q_dist / q_dist.sum()

    return jensenshannon(p_dist, q_dist)


### Runner Helpers ###

def run_on_main_with_unwrapped_network(runner, fn, *args, **kwargs):
    if not runner.is_main_process:
        return None
    network = runner.network
    runner.network = unwrap_model(network)
    try:
        return fn(*args, **kwargs)
    finally:
        runner.network = network


### Geometry ###

def uniform_triangles_sample(triangles):
    tri_origins = triangles[:, 0]
    edges = triangles[:, 1:] - tri_origins[:, None, :]
    uv = np.random.random((len(triangles), 2))
    reflect = uv.sum(axis=1) > 1.0
    uv[reflect] = 1.0 - uv[reflect]
    return tri_origins + uv[:, :1] * edges[:, 0] + uv[:, 1:] * edges[:, 1]


@torch.no_grad()
def Kabsch(x, xref):
    assert isinstance(x, torch.Tensor), 'Input x is not a torch tensor'

    xref = xref.to(x)
    xref = xref - xref.mean(0, keepdim=True)
    b = x.mean(1, keepdim=True)
    x_notran = x - b

    if not torch.isfinite(x_notran).all():
        logging.warning("Non-finite coordinates encountered during Kabsch alignment.")

    xtmp = x_notran.permute((0, 2, 1))
    prod = torch.matmul(xtmp, xref)

    try:
        u, _, vh = torch.linalg.svd(prod)
    except torch._C._LinAlgError:
        error_idx = torch.where(~torch.isfinite(prod))[0].unique()
        logging.warning(f"Non-finite covariance at indices {error_idx}; replacing with zeros before SVD.")
        prod = torch.nan_to_num(prod, nan=0.0, posinf=0.0, neginf=0.0)
        u, _, vh = torch.linalg.svd(prod)

    diag_mat = torch.eye(3, device=x.device, dtype=u.dtype).expand(x.size(0), -1, -1).clone()
    sign_vec = torch.sign(torch.linalg.det(torch.matmul(u, vh))).detach()
    diag_mat[:, 2, 2] = sign_vec

    R = torch.bmm(torch.bmm(u, diag_mat), vh)
    return R.transpose(1, 2), b


def get_RMSD(xvec, xref):
    R, b = Kabsch(xvec, xref)
    b0 = torch.mean(xref, 0, True)
    error = xvec - b - torch.matmul(xref - b0, R)
    return torch.sqrt(torch.sum(error ** 2, dim=(1, 2)) / xref.shape[0])


### Etc ###

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
