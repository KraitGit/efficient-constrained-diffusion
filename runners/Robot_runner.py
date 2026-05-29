import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from sklearn.decomposition import PCA
from scipy.spatial.distance import jensenshannon

from runners.Basic_runner import BasicRunner
from manifolds.Robot import forward_kinematics_pytorch_batched
from src.utils import (
    split_dataset,
    check_memory,
    sample_prior,
    log_constraint_metrics,
    get_constraint_metrics,
    log_validation_summary,
)

class RobotPrior:
    def __init__(self, prior_sample):
        self.prior_sample = torch.from_numpy(prior_sample).float()
    def prior_sampler(self, n):
        indices = torch.randint(0, self.prior_sample.shape[0], (n,))
        return self.prior_sample[indices]

class RobotRunner(BasicRunner):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.load_data()

        self.pca = PCA(n_components=2)
        self.pca.fit(self.full_dataset_features.cpu().numpy())
        self.full_dataset_pca = self.pca.transform(self.full_dataset_features.cpu().numpy())

        if not self.is_main_process:
            return

        logging.info("Visualizing initial training set and forward process.")
        self.plot_sample_hist(self.training_set[:self.config.sample.sample_num].cpu().numpy(), savefig="training_set")
        if self.config.problem.prior_mode == 'load' and self.prior is not None:
            self.plot_sample_hist(self.prior.prior_sample[:self.config.sample.sample_num].cpu().numpy(), savefig="prior_dist")
        
        if self.config.if_train or self.config.if_sample:
            x_hist = self.training_set_path[:self.config.sample.sample_num].clone().transpose(0, 1)
            x = x_hist[-1].cpu().numpy()
            if self.config.problem.if_plot_fwd:
                self.plot_sample_hist(x, savefig='forward_end')

    def load_data(self):
        data_path = './data/robot_arm/'
        paths = np.load(f'{data_path}robot_7dof_joints_T_{self.config.problem.time_steps}_paths.npy')
        labels = np.load(f'{data_path}robot_7dof_joints_T_{self.config.problem.time_steps}_labels.npy')

        data_ori = torch.tensor(paths, dtype=torch.float32).reshape(paths.shape[0], -1)
        labels_ori = torch.tensor(labels, dtype=torch.float32).reshape(labels.shape[0], 1)
        full_dataset = torch.cat([data_ori, labels_ori], dim=1)
        full_dataset = full_dataset[torch.randperm(full_dataset.shape[0])]
        self.full_dataset = full_dataset.clone()

        self.training_set, self.test_set, self.val_set = split_dataset(full_dataset, self.config.seed)
        self.training_labels = self.training_set[:, -1:]
        self.training_set = self.training_set[:, :-1]
        self.test_labels = self.test_set[:, -1:]
        self.test_set = self.test_set[:, :-1]
        self.val_labels = self.val_set[:, -1:]
        self.val_set = self.val_set[:, :-1]
        self.full_dataset_features = self.full_dataset[:, :-1]

        self.prior = None
        if self.config.problem.prior_mode == 'load':
            logging.info("Using 'load' mode for prior. Loading from file.")
            prior_paths = f'{data_path}robot_7dof_joints_prior.npy'
            prior_sample_np = np.load(prior_paths).reshape(-1, self.config.problem.time_steps * 14)
            self.prior = RobotPrior(prior_sample=prior_sample_np)

        elif self.config.problem.prior_mode == 'generate':
            logging.info("Using 'generate' mode...")
        else:
            raise ValueError(f"Invalid prior_mode: {self.config.problem.prior_mode}.")

        if self.config.if_train or self.config.if_sample:
            logging.info("Generating forward path dataset...")
            self.training_set_path, _ = self.generate_path_dataset(self.training_set, keep_quiet=False)
            check_memory(self.training_set_path)

    def _get_initial_samples(self, n, labels):
        if self.config.problem.prior_mode == 'load':
            return self.prior.prior_sampler(n).to(self.device)
        elif self.config.problem.prior_mode == 'generate':
            indices = torch.randint(0, self.training_set.shape[0], (n,))
            init_from_data = self.training_set[indices].to(self.device)
            forward_x, _, _ = self.SDE_sampler_manifolds(
                self.sde,
                self.manifold,
                init_from_data,
                reverse=False,
                score_net=None,
                keep_quiet=True,
                **self.sde_kwargs,
            )
            return forward_x

    def _get_3d_path(self, trajectory_14d_flat):
        """Converts a flattened 14D (cos, sin) trajectory to a 3D end-effector path."""
        if isinstance(trajectory_14d_flat, np.ndarray):
            trajectory_14d_flat = torch.from_numpy(trajectory_14d_flat).float()
        
        traj_flat = trajectory_14d_flat.to(self.device).flatten()
        expected_dim = self.config.problem.time_steps * 14

        # Some datasets keep the class label as the last element; drop extras if present.
        if traj_flat.numel() > expected_dim:
            traj_flat = traj_flat[:expected_dim]
        elif traj_flat.numel() < expected_dim:
            raise ValueError(
                f"Trajectory length {traj_flat.numel()} is shorter than expected {expected_dim} "
                "for converting to 3D path."
            )

        q_cos_sin = traj_flat.reshape(self.config.problem.time_steps, 14)
        
        cos_part = q_cos_sin[..., :7]
        sin_part = q_cos_sin[..., 7:]
        q_points_theta = torch.arctan2(sin_part, cos_part)
        
        with torch.no_grad():
            link_positions = forward_kinematics_pytorch_batched(q_points_theta)
            path_3d = link_positions[self.manifold.end_effector_link_index]
        
        return path_3d.cpu().numpy()

    def plot_topdown_comparison(self, true_samples, gen_samples, savefig=None):
        """
        Plot Top-Down Trajectory View comparing true vs generated trajectories.
        """
        fig_path = f"{self.savefig_dir}/TopDown_{savefig}.pdf"
        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        for traj_flat in true_samples:
            path_3d = self._get_3d_path(traj_flat)

            ax.plot(path_3d[:, 0], path_3d[:, 1], color="lightgreen", alpha=0.3)

        for traj_flat in gen_samples:
            path_3d = self._get_3d_path(traj_flat)
            ax.plot(path_3d[:, 0], path_3d[:, 1], color="salmon", alpha=0.3)

        if self.config.problem.get("obstacles_info"):
            for obs in self.config.problem.obstacles_info:
                ax.add_patch(Circle((obs["position"][0], obs["position"][1]),
                                    radius=0.1, color="black"))

        ax.set_xlim(-0.2, 1.0)
        ax.set_title("Top-Down Trajectory View", fontsize=18, pad=15)
        ax.set_xlabel("X position", fontsize=16)
        ax.set_ylabel("Y position", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.grid(True, alpha=0.4)
        ax.set_aspect("equal", adjustable="box")

        legend_elements = [
            Line2D([0], [0], color="lightgreen", lw=2, label="True Trajectories"),
            Line2D([0], [0], color="salmon", lw=2, label="Generated Trajectories"),
            Circle((0, 0), 0.1, color="black", label="Obstacles"),
        ]
        ax.legend(handles=legend_elements, fontsize=14, loc="upper right", frameon=False)

        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()


    def plot_sample_hist(self, samples, savefig=None):
        fig_path = f"{self.savefig_dir}/Hist_{savefig}.pdf"

        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        for traj_flat in samples:
            path_3d = self._get_3d_path(traj_flat)
            midpoint_x = path_3d[len(path_3d) // 2, 0]
            color = "gold" if midpoint_x > 0.4 else "magenta"
            ax.plot(path_3d[:, 0], path_3d[:, 1], color=color, alpha=0.2)

        if self.config.problem.get("obstacles_info"):
            for obs in self.config.problem.obstacles_info:
                ax.add_patch(Circle((obs["position"][0], obs["position"][1]),
                                    radius=0.1, color="green"))

        ax.set_title(f"Top-Down Trajectory View ({len(samples)} samples)", fontsize=18, pad=15)
        ax.set_xlabel("X position", fontsize=16)
        ax.set_ylabel("Y position", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.grid(True, alpha=0.4)
        ax.set_aspect("equal", adjustable="box")

        legend_elements = [
            Line2D([0], [0], color="magenta", lw=2, label="c=0"),
            Line2D([0], [0], color="gold", lw=2, label="c=1"),
            Circle((0, 0), 0.1, color="green", label="Obstacles"),
        ]
        ax.legend(handles=legend_elements, fontsize=14, loc="upper right", frameon=False)

        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()

    def plot_pca_comparison(self, samples, savefig=None):
        if not hasattr(self, 'pca'):
            return
        samples_np = samples.detach().cpu().numpy() if isinstance(samples, torch.Tensor) else samples
        samples_pca = self.pca.transform(samples_np)
        true_data_pca = self.pca.transform(self.full_dataset_features.cpu().numpy())
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(true_data_pca[:, 0], true_data_pca[:, 1], s=15, alpha=0.3, label='True Data Projection', c='blue')
        ax.scatter(samples_pca[:, 0], samples_pca[:, 1], s=15, alpha=0.5, label='Generated Data Projection', c='red')
        ax.set_title('PCA Projection Comparison', fontsize=18)
        ax.set_xlabel('PC 1', fontsize=14)
        ax.set_ylabel('PC 2', fontsize=14)
        ax.grid(True, linestyle='--')
        ax.legend(fontsize=14)
        plt.savefig(self.savefig_dir + f"/PCA_Plot_{savefig}.pdf", dpi=300)
        plt.close(fig)

    def _compute_jsd_between_histograms(self, generated_hist, reference_hist):
        epsilon = 1e-12
        gen = generated_hist.flatten() + epsilon
        ref = reference_hist.flatten() + epsilon
        gen = gen / gen.sum()
        ref = ref / ref.sum()
        return jensenshannon(gen, ref)

    def _compute_pca_jsd(self, generated_samples, bins=60):
        if not hasattr(self, 'pca'):
            logging.warning("PCA or reference projection missing; skipping JSD computation.")
            return None

        gen_proj = self.pca.transform(generated_samples)
        ref_proj = self.full_dataset_pca

        min_vals = ref_proj.min(axis=0)
        max_vals = ref_proj.max(axis=0)
        span = np.maximum(max_vals - min_vals, 1e-6)
        min_vals = min_vals - span * 0.2
        max_vals = max_vals + span * 0.2

        x_edges = np.linspace(min_vals[0], max_vals[0], bins + 1)
        y_edges = np.linspace(min_vals[1], max_vals[1], bins + 1)

        gen_clipped = np.clip(gen_proj, min_vals, max_vals)
        ref_clipped = np.clip(ref_proj, min_vals, max_vals)

        gen_hist, _, _ = np.histogram2d(gen_clipped[:, 0], gen_clipped[:, 1], bins=[x_edges, y_edges], density=True)
        ref_hist, _, _ = np.histogram2d(ref_clipped[:, 0], ref_clipped[:, 1], bins=[x_edges, y_edges], density=True)
        return self._compute_jsd_between_histograms(gen_hist, ref_hist)

    def validate(self, mode=None, epoch=0, **kwargs):
        if mode == 'start' or mode == 'end':
            return
        prefix = f"val_epoch_{epoch}"
        val_indices = torch.randint(0, len(self.val_set), (self.config.sample.sample_num,))
        labels = self.val_labels[val_indices].to(self.device)
        init = sample_prior(
            self.config.sample.sample_num,
            lambda n: self._get_initial_samples(n, labels),
            self.device,
        )
        x, _, _ = self.sample_backward(init, keep_quiet=True, labels=labels)
        constraints = get_constraint_metrics(self.manifold, x)
        x_np = x.cpu().numpy()
        self.plot_sample_hist(x_np, savefig=f'val_{epoch}_generated')
        self.plot_pca_comparison(x_np, savefig=f'val_{epoch}_generated')
        self.plot_topdown_comparison(self.full_dataset, x_np, savefig=f'val_{epoch}_comparison')

        jsd = self._compute_pca_jsd(x_np)
        metrics = [("JSD on PCA hist.", jsd)] if jsd is not None else []
        log_validation_summary(prefix, self, constraints, metrics)

    def sample_on_manifolds(self):
        logging.info('Start sampling on manifolds.')
        if self.network is not None:
            self.network.to(self.device)
        labels_c0 = torch.zeros(self.config.sample.sample_num // 2, 1, device=self.device)
        init_c0 = sample_prior(
            len(labels_c0),
            lambda n: self._get_initial_samples(n, labels_c0),
            self.device,
        )
        x_c0, x_hist_c0, _ = self.sample_backward(init_c0, keep_quiet=False, labels=labels_c0)
        labels_c1 = torch.ones(self.config.sample.sample_num // 2, 1, device=self.device)
        init_c1 = sample_prior(
            len(labels_c1),
            lambda n: self._get_initial_samples(n, labels_c1),
            self.device,
        )
        x_c1, x_hist_c1, _ = self.sample_backward(init_c1, keep_quiet=False, labels=labels_c1)
        x = torch.cat([x_c0, x_c1], dim=0)
        log_constraint_metrics(self.manifold, x, prefix="backward_sampling")
        x_np, x_hist = x.cpu().numpy(), torch.cat([x_hist_c0, x_hist_c1], dim=1).cpu().numpy()
        self.plot_sample_hist(x_np, savefig='generated_final')
        self.plot_pca_comparison(x_np, savefig='generated_final') 
        plot_idx = list(range(0, 100, 10)) + list(range(90, 101))
        for i in range(self.sde.N + 1):
            if (100 * i / self.sde.N in plot_idx) or (i > self.sde.N - 5):
                self.plot_sample_hist(x_hist[i], savefig=f'generating_bwd_{i}')
        np.save(f"{self.samples_dir}/{self.dataset_name}_samples_generated.npy", x_np)
        np.save(f"{self.samples_dir}/{self.dataset_name}_hist_bwd.npy", x_hist)
