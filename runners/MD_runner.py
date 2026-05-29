import matplotlib.pyplot as plt
import torch
import numpy as np
import logging
from runners.Basic_runner import BasicRunner
from src.utils import (
    split_dataset,
    check_memory,
    get_RMSD,
    sample_prior,
    log_constraint_metrics,
    get_constraint_metrics,
    log_validation_summary,
)
from scipy.spatial.distance import jensenshannon


class MDRunner(BasicRunner):
    def __init__(self, config):
        super().__init__(config)
        # Force zero drift for MD tasks regardless of config drift_mode.
        self.sde.func_b = lambda x: torch.zeros_like(x)
        self.psi_range = (120.0, 180.0)
        self.rmsd_range = (-0.05, 0.6)
        self.jsd_bins = 60

        self.load_data()

        phi_ref = self.manifold.angle_phi(self.xref.unsqueeze(0))/ torch.pi * 180
        self.psi_ref = self.manifold.angle_psi(self.xref.unsqueeze(0))/ torch.pi * 180
        logging.info(f"x reference: phi: {phi_ref.item():.2f}, psi: {self.psi_ref.item():.2f}.")

        if not self.is_main_process:
            return

        samples_test = self.training_set[:self.config.sample.sample_num].detach()
        self.plot_sample_hist(samples_test.cpu().numpy(), savefig="training_set")
        prior_samples = self._get_initial_samples(self.config.sample.sample_num).cpu().numpy()
        self.plot_sample_hist(prior_samples, savefig="prior_dist")

        if self.config.if_train or self.config.if_sample:
            x_hist = self.training_set_path[:self.config.sample.sample_num].clone().transpose(0,1)
            x = x_hist[-1].cpu().numpy()
            self.plot_angle_and_RMSD_hist(x, savefig='forward_end')
            plot_idx = list(range(10)) + list(range(10, 101, 10))
            for i in range(self.sde.N+1):
                if (100 * i / self.sde.N in plot_idx) or (i < 5):
                    x_temp = x_hist[i].cpu().numpy()
                    self.plot_angle_and_RMSD_hist(x_temp, savefig=f'generating_fwd_{i}')
            np.save(f"{self.samples_dir}/{self.dataset_name}_hist_fwd.npy", x_hist.cpu().detach().numpy())

        return

    def load_data(self):
        """
        Loads the dipeptide datasets for the region around psi=150 degrees.
        """
        self.kappa = self.config.training.kappa
        data_path = './data/dipeptide/'

        ref_path = f'{data_path}dipeptide_ref_phi_psiwin.npy'
        self.xref = torch.tensor(np.load(ref_path)).float()

        center_path = f'{data_path}dipeptide_center.npy'
        self.x_center = torch.tensor(np.load(center_path)).float()

        dataset_path = f'{data_path}dipeptide_refined_phi_psiwin.npy'
        data_ori = torch.tensor(np.load(dataset_path)).float()

        self.natom = 10

        self.data_set = data_ori[torch.randperm(data_ori.shape[0])].clone()
        self.training_set, self.test_set, self.val_set = split_dataset(self.data_set, self.config.seed)

        self.training_set = self.training_set.reshape(-1, self.natom * 3)
        self.test_set = self.test_set.reshape(-1, self.natom * 3)
        self.val_set = self.val_set.reshape(-1, self.natom * 3)
        reference_data = torch.cat([self.training_set, self.val_set, self.test_set], dim=0).reshape(-1, self.natom, 3)
        self.reference_psi_angles = (
            self.manifold.angle_psi(reference_data) / torch.pi * 180
        ).cpu().numpy().reshape(-1)
        self.reference_rmsd_to_center = get_RMSD(reference_data, self.x_center).cpu().numpy()

        if self.config.if_train or self.config.if_sample:
            self.training_set_path = self.generate_path_dataset(self.training_set, keep_quiet=False)[0]
            check_memory(self.training_set_path)

    def plot_sample_hist(self, samples, savefig=None):  
        samples = torch.tensor(samples).reshape(-1, self.natom, 3).float()

        phi, psi = self.manifold.angle_phi(samples), self.manifold.angle_psi(samples)
        phi_deg = (phi.reshape(-1) / torch.pi * 180).detach().cpu().numpy()
        psi_deg = (psi.reshape(-1) / torch.pi * 180).detach().cpu().numpy()

        phi_deg = phi_deg[np.isfinite(phi_deg)]
        psi_deg = psi_deg[np.isfinite(psi_deg)]
        phi_deg = np.clip(phi_deg, -180.0, 180.0)
        psi_deg = np.clip(psi_deg, -180.0, 180.0)

        fig = plt.figure(figsize=(8, 4))
        n_valid = min(len(phi_deg), len(psi_deg))
        bins = int(n_valid / 100) if n_valid >= 100 else 10
        bins = max(1, bins)
        phi_edges = np.linspace(-180.0, 180.0, bins + 1)
        psi_edges = np.linspace(-180.0, 180.0, bins + 1)

        ax = plt.subplot(1, 2, 1)
        if len(phi_deg) > 0:
            ax.hist(phi_deg, bins=phi_edges, alpha=1.0, density=True, color="green")
        else:
            ax.text(0.5, 0.5, "no finite phi", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("phi angle", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_xlim(-180, 180)

        ax = plt.subplot(1, 2, 2)
        if len(psi_deg) > 0:
            ax.hist(psi_deg, bins=psi_edges, alpha=1.0, density=True, color="blue")
        else:
            ax.text(0.5, 0.5, "no finite psi", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("psi angle", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_xlim(120, 180)

        psi_ref = self.psi_ref.reshape(-1).numpy()
        ax.axvline(x=psi_ref, color='red', linestyle='--')

        if hasattr(self.manifold, 'psi_windows_rad') and self.manifold.l > 0:
            psi_windows_deg = np.rad2deg(self.manifold.psi_windows_rad.numpy())
            for (low, high) in psi_windows_deg:
                ax.axvspan(low, high, color='blue', alpha=0.1, zorder=-1)
                ax.axvline(x=low, color='blue', linestyle=':', linewidth=1.5)
                ax.axvline(x=high, color='blue', linestyle=':', linewidth=1.5)

        plt.savefig(self.savefig_dir + f"/Hist_psi_phi_{savefig}.pdf",
                    bbox_inches='tight')
        plt.close(fig)

    def plot_angle_and_RMSD_hist(self, samples, savefig=None):
        if not isinstance(samples, torch.Tensor):
            samples = torch.tensor(samples).float()
        true_set = self.training_set.detach()
        
        samples = samples.reshape(-1, self.natom, 3)
        true_set = true_set.reshape(-1, self.natom, 3)

        psi = self.manifold.angle_psi(samples)
        psi_true = self.manifold.angle_psi(true_set)
        psi_angle = psi.reshape(-1).numpy() / np.pi * 180
        psi_angle_true = psi_true.reshape(-1).numpy() / np.pi * 180

        psi_center = self.manifold.angle_psi(self.x_center.unsqueeze(0)) / torch.pi * 180

        RMSD = get_RMSD(samples, self.x_center).numpy()
        RMSD_true = get_RMSD(true_set, self.x_center).numpy()

        psi_angle = psi_angle[np.isfinite(psi_angle)]
        psi_angle_true = psi_angle_true[np.isfinite(psi_angle_true)]
        RMSD = RMSD[np.isfinite(RMSD)]
        RMSD_true = RMSD_true[np.isfinite(RMSD_true)]

        psi_angle = np.clip(psi_angle, -180.0, 180.0)
        psi_angle_true = np.clip(psi_angle_true, -180.0, 180.0)
        RMSD = np.clip(RMSD, self.rmsd_range[0], self.rmsd_range[1])
        RMSD_true = np.clip(RMSD_true, self.rmsd_range[0], self.rmsd_range[1])

        fig = plt.figure(figsize=(8, 4))
        n_valid = min(len(psi_angle), len(psi_angle_true), len(RMSD), len(RMSD_true))
        bins = int(n_valid / 100) if n_valid >= 100 else 10
        bins = max(1, bins)
        psi_edges = np.linspace(-180.0, 180.0, bins + 1)
        rmsd_edges = np.linspace(self.rmsd_range[0], self.rmsd_range[1], bins + 1)

        ax = plt.subplot(1, 2, 1)
        if len(psi_angle) > 0:
            ax.hist(psi_angle, bins=psi_edges, alpha=0.5, density=True, color='green', label='Generated')
        if len(psi_angle_true) > 0:
            ax.hist(psi_angle_true, bins=psi_edges, alpha=0.5, density=True, color='red', label='True')
        ax.axvline(x=psi_center.reshape(-1).numpy(), color='black', linestyle='--')
        ax.set_title(r'$\psi$ distribution', fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_xlim(120, 180)

        if hasattr(self.manifold, 'psi_windows_rad') and self.manifold.l > 0:
            psi_windows_deg = np.rad2deg(self.manifold.psi_windows_rad.numpy())
            for (low, high) in psi_windows_deg:
                ax.axvspan(low, high, color='blue', alpha=0.1, zorder=-1)
                ax.axvline(x=low, color='blue', linestyle=':', linewidth=1.5)
                ax.axvline(x=high, color='blue', linestyle=':', linewidth=1.5)

        ax = plt.subplot(1, 2, 2)
        if len(RMSD) > 0:
            ax.hist(RMSD, bins=rmsd_edges, alpha=0.5, density=True, color='green', label='Generated')
        if len(RMSD_true) > 0:
            ax.hist(RMSD_true, bins=rmsd_edges, alpha=0.5, density=True, color='red', label='True')
        ax.set_title(r"RMSD to Center ($\psi\approx150^\circ$)", fontsize=16)       
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_xlim(*self.rmsd_range)
        ax.legend(fontsize=9)

        plt.savefig(self.savefig_dir + f"/Hist_angel_RMSD_{savefig}.pdf",
                    bbox_inches='tight')
        plt.close(fig)

    def _compute_jsd_from_histograms(self, generated, reference, value_range, bins=None):
        if generated.size == 0 or reference.size == 0:
            return None
        bins = bins or self.jsd_bins
        vmin, vmax = value_range
        if vmax <= vmin:
            vmax = vmin + 1e-6
        edges = np.linspace(vmin, vmax, bins + 1)

        # Clamp outliers to nearest bin so every sample is accounted for.
        gen_clipped = np.clip(generated, vmin, vmax)
        ref_clipped = np.clip(reference, vmin, vmax)

        gen_hist, _ = np.histogram(gen_clipped, bins=edges, density=True)
        ref_hist, _ = np.histogram(ref_clipped, bins=edges, density=True)

        epsilon = 1e-6
        gen = gen_hist + epsilon
        ref = ref_hist + epsilon
        gen = gen / gen.sum()
        ref = ref / ref.sum()
        return jensenshannon(gen, ref)

    def _compute_psi_and_rmsd(self, samples_np):
        samples_tensor = torch.tensor(samples_np, dtype=torch.float32).reshape(-1, self.natom, 3)
        psi_angles = (self.manifold.angle_psi(samples_tensor) / torch.pi * 180).cpu().numpy().reshape(-1)
        rmsd = get_RMSD(samples_tensor, self.x_center).cpu().numpy()
        return psi_angles, rmsd


    def _get_initial_samples(self, n: int):
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


    def validate(self, mode=None, epoch=0, **kwargs):
        if mode == 'start' or mode == 'end':
            return

        prefix = f"val_epoch_{epoch}"
        init = sample_prior(
            self.config.sample.sample_num,
            lambda n: self._get_initial_samples(n),
            self.device,
        )
        x, _, _ = self.sample_backward(init, keep_quiet=True)
        constraints = get_constraint_metrics(self.manifold, x)
        x = x.cpu().numpy()
        self.plot_angle_and_RMSD_hist(x, savefig=f'val_{epoch}_generated')
        psi_gen, rmsd_gen = self._compute_psi_and_rmsd(x)
        psi_jsd = self._compute_jsd_from_histograms(
            psi_gen, self.reference_psi_angles, self.psi_range, bins=self.jsd_bins
        )
        rmsd_jsd = self._compute_jsd_from_histograms(
            rmsd_gen, self.reference_rmsd_to_center, self.rmsd_range, bins=self.jsd_bins
        )
        metrics = []
        if psi_jsd is not None:
            metrics.append(("JSD on psi hist.", psi_jsd))
        if rmsd_jsd is not None:
            metrics.append(("JSD on RMSD hist.", rmsd_jsd))
        log_validation_summary(
            prefix,
            self,
            constraints,
            metrics,
        )

    def sample_on_manifolds(self):
        logging.info(f'Start sampling on manifolds.')
        device = self.device
        if self.network is not None: self.network.to(device)

        logging.info("Start sampling backward SDE.")
        init = sample_prior(
            self.config.sample.sample_num,
            lambda n: self._get_initial_samples(n),
            self.device,
        )

        x, x_hist, _ = self.sample_backward(init, keep_quiet=False)
        log_constraint_metrics(self.manifold, x, prefix="backward_sampling")
        self.calculate_constraint(x)
        x = x.cpu().numpy()
        self.plot_angle_and_RMSD_hist(x, savefig='generated')
        
        plot_idx = list(range(0, 100, 10)) + list(range(90, 101))
        for i in range(self.sde.N+1):
            if (100 * i / self.sde.N in plot_idx) or (i > self.sde.N - 5):
                x_temp = x_hist[i].cpu().numpy()
                self.plot_angle_and_RMSD_hist(x_temp, savefig=f'generating_bwd_{i}')

        np.save(f"{self.samples_dir}/{self.dataset_name}_samples_generated.npy", x)
        np.save(f"{self.samples_dir}/{self.dataset_name}_hist_bwd.npy", x_hist.cpu().detach().numpy())

        return
