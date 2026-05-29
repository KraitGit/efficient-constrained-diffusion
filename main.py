import os
import torch
import torch.distributed as dist
import hydra

from omegaconf import DictConfig, OmegaConf, open_dict
from src.utils import get_logger, select_device, set_seed_everywhere, setup_distributed
import runners


@hydra.main(version_base=None, config_path="configs", config_name="main") 
def main(config: DictConfig):
    distributed, rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    set_seed_everywhere(config.seed)

    if distributed:
        now_value = [str(config.now) if is_main else None]
        dist.broadcast_object_list(now_value, src=0)
        config.now = now_value[0]

    workdir = os.path.join('results', config.problem.manifold,
                           config.problem.dataset,
                           config.save_prefix + f"-{config.seed}-" + config.now + f"-{config.sample.sampler}")
    with open_dict(config):
        config.workdir = workdir
        config.distributed = distributed
        config.rank = rank
        config.world_size = world_size
        config.local_rank = local_rank

    log_dir = os.path.join(config.workdir, 'logs')
    if distributed and not is_main:
        dist.barrier()
    logging = get_logger(log_dir, rank=rank)
    if distributed and is_main:
        dist.barrier()

    device, gpu = select_device(config.gpu, distributed=distributed, local_rank=local_rank)

    with open_dict(config):
        config.gpu = gpu
        config.device = device

    if is_main:
        logging.info(f"Found {os.cpu_count()} total number of CPUs.")
    if is_main and torch.cuda.is_available():
        logging.info(f"Found {torch.cuda.device_count()} CUDA devices.")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logging.info(f"{props.name} with Memory: {props.total_memory / (1024 ** 3):.2f}GB")
    logging.info(f"Using device: {config.device} (rank {rank}/{world_size})")

    if is_main:
        yaml_str = OmegaConf.to_yaml(config)
        with open(os.path.join(log_dir, 'config.yml'), 'w') as file:
            file.write(yaml_str)
        logging.info(f"Writing log file to {log_dir}")
        logging.info(">" * 80)
        logging.info(yaml_str)
        logging.info("<" * 80)

    try:
        runner = getattr(runners, config.problem.runner)(config)
        runner.run()
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
