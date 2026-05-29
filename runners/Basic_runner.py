import matplotlib
matplotlib.use('Agg')
import logging
import models
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import math
import manifolds
import time


from src.sampling import (
    SDE_sampler_manifolds_OLLA_P,
    SDE_sampler_manifolds_OLLA,
    SDE_sampler_manifolds_ULLA_P,
    SDE_sampler_manifolds_ULLA,
)
from src.sde_lib import SDE_Brownian_manifolds
from src.utils import (
    ExponentialMovingAverage,
    save_model,
    load_model,
    check_memory,
    synchronize_device,
    distributed_barrier,
    unwrap_model,
    model_parameters,
    maybe_wrap_ddp,
    reduce_mean_float,
    make_index_loader,
    log_runner_sampling_timing,
    run_on_main_with_unwrapped_network,
)
from src.loss_utils import loss_overdamped_path, loss_underdamped_path


class BasicRunner:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.distributed = bool(getattr(config, 'distributed', False))
        self.rank = int(getattr(config, 'rank', 0))
        self.world_size = int(getattr(config, 'world_size', 1))
        self.local_rank = getattr(config, 'local_rank', None)
        self.is_main_process = self.rank == 0

        self.workdir = self.config.workdir
        self.savefig_dir = os.path.join(self.workdir, 'figs')
        self.samples_dir = os.path.join(self.workdir, 'samples')
        self.validate_dir = os.path.join(self.workdir, 'validate')
        os.makedirs(self.savefig_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.validate_dir, exist_ok=True)
        self.dataset_name = self.config.problem.dataset
        self.training_finished = False

        self.trajectory_gen_time = 0.0
        self.trajectory_sample_count = 0
        self.training_time_total = 0.0
        self.training_sample_count = 0
        self.path_dataset_is_sharded = False

        if self.config.problem.manifold == 'S2':
            self.manifold = manifolds.Manifold_Sphere(dim=2)
        elif self.config.problem.manifold == "SOn":
            self.manifold = manifolds.Manifold_SOn(self.config.problem.mat_dim)
        elif self.config.problem.manifold == "SDF":
            self.obj = self.dataset_name.split('_')[0]
            self.sdf_model_path = f'./constraint/model/{self.obj}_whole_sdf.pt'
            mesh_path = f"./data/{self.obj}/{self.obj}_mesh_simple.ply"
            self.manifold = manifolds.Manifold_SDF(model_path=self.sdf_model_path, mesh_path=mesh_path)
            if self.is_main_process:
                torch.save(self.manifold.model.state_dict(), os.path.join(self.workdir + "/sdf_constraint_model.pt"))
            distributed_barrier(self.distributed)
            self.manifold.model.to(self.device)
        elif self.config.problem.manifold == "MD":
            self.manifold = manifolds.Manifold_MD(psi_windows=[(self.config.problem.psi_windows_low,
                                                   self.config.problem.psi_windows_high)],
                                                   boundary_repulsion=self.config.sample.epsilon)
        elif self.config.problem.manifold == "Robot":
            self.manifold = manifolds.Manifold_Robot(time_steps=self.config.problem.time_steps,
                                                     target_ee_z=self.config.problem.target_ee_z,
                                                     obstacles_info=self.config.problem.obstacles_info,
                                                     safety_margin=self.config.problem.safety_margin,
                                                     obstacle_radius=self.config.problem.obstacle_radius,
                                                     boundary_repulsion_rate=self.config.sample.epsilon)

        else:
            raise NotImplementedError(f"Manifold {self.config.problem.manifold} is not implemented.")

        sampler = self.config.sample.sampler
        supported_samplers = {'OLLA-P', 'OLLA', 'ULLA-P', 'ULLA'}
        if sampler not in supported_samplers:
            raise NotImplementedError(
                f"Sampler {sampler} is not implemented. "
                "Supported samplers: OLLA-P, OLLA, ULLA-P, ULLA."
            )

        logging.info(f"Current sampler: {sampler}")
        overdamped_sampler = sampler in ('OLLA-P', 'OLLA')
        sigma_schedule = self.config.model.sigma_schedule_overdamped if overdamped_sampler else self.config.model.sigma_schedule_underdamped
        n_key = 'N_overdamped' if overdamped_sampler else 'N_underdamped'
        n_steps = self.config.model[n_key] if n_key in self.config.model else self.config.model.N
        self.sde = SDE_Brownian_manifolds(
            sigma_min=self.config.model.sigma_min_overdamped if overdamped_sampler else self.config.model.sigma_min_underdamped,
            sigma_max=self.config.model.sigma_max_overdamped if overdamped_sampler else self.config.model.sigma_max_underdamped,
            N=n_steps,
            T=self.config.model.T_overdamped if overdamped_sampler else self.config.model.T_underdamped,
            sampler=sampler,
            drift_mode=getattr(self.config.sample, 'drift_mode', 'zero'),
            sigma_schedule=sigma_schedule
        )
        if sampler == 'OLLA-P':
            self.SDE_sampler_manifolds = SDE_sampler_manifolds_OLLA_P
            self.sde_kwargs = {}
            logging.info("Using OLLA-P sampler")

        elif sampler == 'OLLA':
            self.SDE_sampler_manifolds = SDE_sampler_manifolds_OLLA
            self.sde_kwargs = {
                'alpha': self.config.sample.sampler_OLLA_alpha,
                'projection_mode': getattr(self.config.sample, 'projection_mode_olla', 'implicit'),
            }
            logging.info(f"Using OLLA sampler with alpha = {self.config.sample.sampler_OLLA_alpha}")

        elif sampler == 'ULLA-P':
            self.SDE_sampler_manifolds = SDE_sampler_manifolds_ULLA_P
            logging.info("Using ULLA-P sampler")
            self.sde_kwargs = dict(
                gamma=self._sampler_value('sampler_gamma'),
            )

        elif sampler == 'ULLA':
            self.SDE_sampler_manifolds = SDE_sampler_manifolds_ULLA
            logging.info("Using ULLA sampler")
            self.sde_kwargs = dict(
                gamma=self._sampler_value('sampler_gamma'),
                alpha=self.config.sample.sampler_ULLA_alpha,
                projection_mode=getattr(self.config.sample, 'projection_mode_ulla', 'explicit'),
                terminal_projection=getattr(self.config.sample, 'terminal_projection', True),
                projection_threshold=getattr(self.config.sample, 'projection_threshold', 1e-5),
            )
                                    
    def _sampler_value(self, name):
        sampler_key = self.config.sample.sampler.replace('-', '_')
        specific_name = f'{name}_{sampler_key}'
        if specific_name in self.config.sample:
            return self.config.sample[specific_name]
        return self.config.sample[name]

    def get_network(self):
        network_mode = self.config.training.network_mode
        if network_mode == 'MLP':
            if self.config.sample.sampler in ['ULLA-P', 'ULLA']:
                network_input_dim = 2 * self.manifold.out_dim + 1
            else:
                network_input_dim = self.manifold.out_dim + 1

            layers = [network_input_dim] + self.config.training.hidden_layers + [self.manifold.out_dim]

            network = models.MLP(layers, activation=self.config.training.activation)
            
        elif network_mode == 'EMLP':
            if self.config.sample.sampler in ['ULLA-P', 'ULLA']:
                layers = [6*self.natom+1] + self.config.training.hidden_layers + [3*self.natom]
            else:
                layers = [3*self.natom+1] + self.config.training.hidden_layers + [3*self.natom]
            network = models.EMLP(layers, xref=self.xref, activation=self.config.training.activation)

        elif network_mode == 'TemporalUNet':
            base_state_dim = getattr(self.manifold, 'input_dim', None)
            if base_state_dim is None and hasattr(self.config.problem, 'time_steps'):
                time_steps = self.config.problem.time_steps
                if time_steps > 0 and self.manifold.out_dim % time_steps == 0:
                    base_state_dim = self.manifold.out_dim // time_steps
            if base_state_dim is None:
                base_state_dim = self.manifold.out_dim

            underdamped_samplers = {'ULLA-P', 'ULLA'}
            if self.config.sample.sampler in underdamped_samplers:
                input_state_dim = 2 * base_state_dim
                output_state_dim = base_state_dim
            else:
                input_state_dim = base_state_dim
                output_state_dim = base_state_dim

            layers = [input_state_dim] + self.config.training.hidden_layers + [output_state_dim]
            network = models.TemporalUNet(
                layers,
                input_state_dim=input_state_dim,
                output_state_dim=output_state_dim,
                scale=1.0,
                activation=self.config.training.activation
            )
        else:
            raise NotImplementedError(f"Network mode {network_mode} is not implemented.")
        return network

    def sample_backward(self, init, keep_quiet=False, **kwargs):
        return self.SDE_sampler_manifolds(
            self.sde, self.manifold, init,
            reverse=True,
            score_net=self.network,
            keep_quiet=keep_quiet,
            **self.sde_kwargs,
            **kwargs
        )

    def run(self):
        if self.config.if_train:
            self.network = self.get_network()
            self.training_finished = False
            self.train_step()
        else:
            model_path = os.path.join(self.workdir, self.config.load_model_path)
            self.network = load_model(model_path)
            self.training_finished = True
        if self.is_main_process:
            save_model(self.workdir, self.network, name='model.pt')
        distributed_barrier(self.distributed)

        if self.is_main_process and self.config.if_sample and self.training_finished:
            self.sample_on_manifolds()

    def _build_lr_scheduler(self, optimizer):
        scheduler_cfg = getattr(self.config.optim, 'scheduler', None)
        if scheduler_cfg is None:
            return None

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(getattr(scheduler_cfg, 'mode', 'min')),
            factor=float(getattr(scheduler_cfg, 'factor', 0.5)),
            patience=int(getattr(scheduler_cfg, 'patience', 100)),
            threshold=float(getattr(scheduler_cfg, 'threshold', 1e-4)),
            threshold_mode=str(getattr(scheduler_cfg, 'threshold_mode', 'rel')),
            cooldown=int(getattr(scheduler_cfg, 'cooldown', 0)),
            min_lr=float(getattr(scheduler_cfg, 'min_lr', 1e-6)),
            eps=float(getattr(scheduler_cfg, 'eps', 1e-8))
        )
        extra_cfg = (f"factor={getattr(scheduler_cfg, 'factor', 0.5)}, patience={getattr(scheduler_cfg, 'patience', 100)}, "
                     f"threshold={getattr(scheduler_cfg, 'threshold', 1e-4)}, mode={getattr(scheduler_cfg, 'mode', 'min')}, "
                     f"cooldown={getattr(scheduler_cfg, 'cooldown', 0)}, min_lr={getattr(scheduler_cfg, 'min_lr', 1e-6)}")

        scheduler_state = {
            'instance': scheduler,
            'warmup_epochs': max(0, int(getattr(scheduler_cfg, 'warmup_epochs', 0))),
            'base_lr': float(self.config.optim.lr)
        }
        logging.info(f"Using ReduceLROnPlateau LR scheduler with config: {extra_cfg}, warmup_epochs={scheduler_state['warmup_epochs']}")
        return scheduler_state

    def _step_lr_scheduler(self, optimizer, scheduler_state, epoch, metric=None):
        scheduler = scheduler_state['instance']
        warmup_epochs = scheduler_state['warmup_epochs']
        base_lr = scheduler_state['base_lr']

        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_progress = (epoch + 1) / warmup_epochs
            new_lr = base_lr * warmup_progress
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            return new_lr

        if metric is None:
            raise ValueError("ReduceLROnPlateau scheduler requires a metric (e.g., loss) to step.")
        scheduler.step(metric)
        return optimizer.param_groups[0]['lr']

    def train_step(self):
        if self.is_main_process:
            logging.info('Start training...')
            logging.info(f"Network:\n {self.network.__str__()}")
            logging.info(f"Numbers of parameters: {sum(p.numel() for p in self.network.parameters() if p.requires_grad)}")

        self.network.to(self.device)
        self.network = maybe_wrap_ddp(self.network, self.distributed, self.local_rank)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.optim.lr, weight_decay=0.000, betas=(0.9, 0.999), amsgrad=False)
        scheduler_state = self._build_lr_scheduler(optimizer)
        self.ema = ExponentialMovingAverage(model_parameters(self.network), self.config.optim.ema)

        training_loader, training_sampler = make_index_loader(
            self.training_set_path.shape[0],
            self.config.training.batch_size,
            self.distributed,
            self.world_size,
            self.rank,
            already_sharded=self.path_dataset_is_sharded,
        )
        self.total_epochs = self.config.training.n_epochs
        default_val_freq = max(1, int(self.total_epochs / 20)) if self.total_epochs > 0 else 1
        val_freq = self.config.training.val_freq if self.config.training.val_freq > 0 else default_val_freq
        start_epoch, loss_train_list, step = 0, [], 0

        run_on_main_with_unwrapped_network(self, self.validate, mode='start')
        distributed_barrier(self.distributed)
        for epoch in range(start_epoch, self.total_epochs + 1):
            if training_sampler is not None:
                training_sampler.set_epoch(epoch)
            for i, sample_indices in enumerate(training_loader):
                step += 1

                synchronize_device(self.device)
                iter_start_time = time.time()
                samples = self.training_set_path[sample_indices, :].to(self.device)
                loss = self.loss_fn(samples, epoch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_parameters(self.network), max_norm=10.0)
                optimizer.step()
                synchronize_device(self.device)
                iter_time = time.time() - iter_start_time
                self.training_time_total += iter_time
                self.training_sample_count += max(1, samples.shape[0])

                if epoch > 0:
                    self.ema.update(model_parameters(self.network))
            
            last_loss = reduce_mean_float(loss.detach().cpu().item(), self.device, self.distributed, self.world_size)
            loss_train_list.append(last_loss)
            if scheduler_state is not None:
                self._step_lr_scheduler(optimizer, scheduler_state, epoch, metric=last_loss)

            if epoch % val_freq == 0:
                if self.is_main_process:
                    save_model(self.validate_dir, self.network, name=f"model_temp.pt")
                self.ema.store(model_parameters(self.network))
                self.ema.copy_to(model_parameters(self.network))

                run_on_main_with_unwrapped_network(self, self.validate, epoch=epoch, step=step, batch=samples[:, 0, :self.manifold.out_dim].detach().cpu().clone())
                self.ema.restore(model_parameters(self.network))
                distributed_barrier(self.distributed)

                if self.is_main_process:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    epochs_loss = np.arange(len(loss_train_list))
                    axes[0].plot(epochs_loss, loss_train_list, label='Training Loss')
                    axes[0].set_xlabel('Epoch')
                    axes[0].set_ylabel('Training Loss')
                    axes[0].grid(True)
                    axes[0].legend()

                    if hasattr(self, 'h_val') and len(self.h_val) > 0:
                        h_vals = [h.mean() if isinstance(h, np.ndarray) else h for h in self.h_val]
                        record_step = (self.config.training.update_training_set_path_freq or val_freq)
                        epochs_h = np.arange(len(h_vals)) * record_step
                        axes[1].plot(epochs_h, h_vals, label='Equality Constraint Violation')
                        axes[1].legend()
                    else:
                        axes[1].text(0.5, 0.5, 'no data', ha='center', va='center')
                    axes[1].set_xlabel('Epoch')
                    axes[1].set_ylabel('Equality Constraint Violation')
                    axes[1].grid(True)

                    plt.savefig(self.savefig_dir + f"/aa_loss_constraint_{self.dataset_name}.png",
                                dpi=300, bbox_inches='tight')
                    plt.close(fig)
                
            if self.config.training.update_training_set_path_freq > 0 and epoch % self.config.training.update_training_set_path_freq == 0 and epoch > 0:
                del self.training_set_path
                check_memory(keep_quiet=True)
                self.training_set_path, h_val = self.generate_path_dataset(self.training_set, keep_quiet=True)

                training_loader, training_sampler = make_index_loader(
                    self.training_set_path.shape[0],
                    self.config.training.batch_size,
                    self.distributed,
                    self.world_size,
                    self.rank,
                    already_sharded=self.path_dataset_is_sharded,
                )
                h_mean = reduce_mean_float(h_val.mean().detach().cpu().item(), self.device, self.distributed, self.world_size)
                if self.is_main_process:
                    self.h_val = self.h_val + [h_mean] if hasattr(self, 'h_val') else [h_mean]
        

        run_on_main_with_unwrapped_network(self, self.validate, mode='end')
        self.ema.copy_to(model_parameters(self.network))
        self.network = unwrap_model(self.network)
        self.training_finished = True
        if self.is_main_process:
            logging.info("training complete")

    def loss_fn(self, path_batch, epoch = None):
        data_hist = path_batch.transpose(0, 1).contiguous()
        sigmas = self.sde.sde(None, torch.linspace(0., self.sde.T, self.sde.N+1, device=data_hist.device))[1][:-1]

        if self.config.sample.sampler in ['OLLA-P', 'OLLA']:
            loss = loss_overdamped_path(
                        self.manifold, data_hist,
                        score_net = self.network,
                        func_b    = self.sde.func_b,
                        sigmas    = sigmas,
                        dt        = self.sde.dt,
                        project_to_tangent=True)
        elif self.config.sample.sampler in ['ULLA-P', 'ULLA']:
            loss = loss_underdamped_path(
                        self.manifold, data_hist,
                        score_net = self.network,
                        func_b = self.sde.func_b,
                        sigmas   = sigmas,
                        dt     = self.sde.dt,
                        gamma  = self.config.sample.sampler_gamma,
                        )
        
        else:
            raise NotImplementedError
        return loss

    def generate_path_dataset(self, data_init, keep_quiet=False):
        if not keep_quiet:
            logging.info("-------------------------Start generating path dataset.-------------------------")
        
        sampling_time_total = 0.0
        device = self.device
        if self.distributed:
            original_samples = data_init.shape[0]
            indices = torch.arange(self.rank, original_samples, self.world_size)
            data_init = data_init[indices]
            self.path_dataset_is_sharded = True
            if not keep_quiet:
                logging.info(
                    f"rank {self.rank}: generating {data_init.shape[0]}/{original_samples} training paths"
                )
        else:
            self.path_dataset_is_sharded = False

        total_samples = data_init.shape[0]
        chunk_size = getattr(self.config.training, 'chunk_batch_size', total_samples)
        chunk_size = total_samples if chunk_size is None else int(chunk_size)
        chunk_size = max(1, min(chunk_size, total_samples))
        num_chunks = math.ceil(total_samples / chunk_size)

        data_hist_chunks = []
        h_val_chunks = []

        for chunk_idx, start in enumerate(range(0, total_samples, chunk_size), 1):
            end = min(start + chunk_size, total_samples)
            x_init = data_init[start:end].to(device)

            synchronize_device(device)
            chunk_start_time = time.time()
            _, data_hist, _ = self.SDE_sampler_manifolds(
                self.sde, self.manifold, x_init,
                reverse=False,
                keep_quiet=keep_quiet, **self.sde_kwargs
            )
            synchronize_device(device)
            chunk_time = time.time() - chunk_start_time
            chunk_samples = end - start
            sampling_time_total += chunk_time
            self.trajectory_gen_time += chunk_time
            self.trajectory_sample_count += chunk_samples

            x_hist = data_hist
            x_hist_flat = x_hist.reshape(-1, x_hist.shape[-1])
            h_val_chunk = self.manifold.constraint_fn(x_hist_flat)

            data_hist_chunks.append(data_hist.detach().transpose(0, 1).cpu())
            h_val_chunks.append(h_val_chunk.detach().cpu())

            if not keep_quiet:
                logging.info(
                    f"sampling_chunk: chunk={chunk_idx}/{num_chunks}, samples={chunk_samples}, "
                    f"time={chunk_time:.2f}s, per_sample={chunk_time / chunk_samples:.6f}s/sample"
                )

            del data_hist, x_hist, x_hist_flat, h_val_chunk

        if not keep_quiet:
            logging.info(
                f"sampling_time: total={sampling_time_total:.2f}s, "
                f"per_sample={sampling_time_total / total_samples:.6f}s/sample, samples={total_samples}"
            )
            log_runner_sampling_timing(self, "cumulative")

        data_hist_full = torch.cat(data_hist_chunks, dim=0)
        h_val_full = torch.cat(h_val_chunks, dim=0)
        if self.distributed:
            local_count = torch.tensor(data_hist_full.shape[0], device=device)
            torch.distributed.all_reduce(local_count, op=torch.distributed.ReduceOp.MIN)
            min_count = int(local_count.item())
            data_hist_full = data_hist_full[:min_count]
        return data_hist_full, h_val_full

    def calculate_constraint(self, samples):
        sample_constraints = self.manifold.constraint_fn(samples)
        logging.info(f'generated constraint_range: min={sample_constraints.min().item():.6f}, max={sample_constraints.max().item():.6f}')
