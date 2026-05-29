import logging
import os
import matplotlib.pyplot as plt
import torch
import numpy as np

import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from runners.Basic_runner import BasicRunner
from manifolds.Sphere import latlon_to_xyz, xyz_to_latlon
from src.utils import (
    split_dataset,
    check_memory,
    sample_prior,
    log_constraint_metrics,
    get_constraint_metrics,
    log_validation_summary,
    compute_jsd_2d_histogram,
    save_model,
    load_model,
)

class S2Runner(BasicRunner):
    def __init__(self, config):
        super().__init__(config)
        self.load_data()
        if not self.is_main_process:
            return

        x_prior = self.manifold.uniform_sample(self.config.sample.sample_num)
        self.plot_sample(x_prior, savefig='prior')

        x_hist = self.training_set_path.clone().transpose(0, 1)
        plot_idx = list(range(10)) + list(range(10, 101, 10))
        for i in range(self.sde.N+1):
            if (100 * i / self.sde.N in plot_idx) or (i < 5):
                self.plot_sample(x_hist[i].cpu().numpy(), savefig=f'generating_fwd_{i}')

    def load_data(self):
        csv_path = f"./data/S2/earth_data/{self.dataset_name}.csv"
        original_data = pd.read_csv(csv_path, comment='#', header=0).values.astype("float32")
        original_data = latlon_to_xyz(original_data)
        self.config.sample.sample_num = original_data.shape[0]
        self.projection = ccrs.PlateCarree(central_longitude=0)

        original_data = torch.tensor(original_data, dtype=torch.float32)
        self.training_set, self.test_set, self.val_set = split_dataset(original_data, self.config.seed)
        self.reference_latlon = self._to_latlon(original_data)
        self.val_latlon = self._to_latlon(self.val_set)
        self.test_latlon = self._to_latlon(self.test_set)
        self.best_val_jsd = float("inf")
        self.best_val_epoch = None
        self.best_val_path = os.path.join(self.validate_dir, "model_best_val_jsd.pt")

        self.training_set_path, _ = self.generate_path_dataset(self.training_set, keep_quiet=False)
        check_memory(self.training_set_path)

    def _to_latlon(self, samples):
        lat, lon = xyz_to_latlon(samples.detach().cpu().numpy())
        return np.stack([lat, lon], axis=1)

    def _sample_generated_latlon(self):
        init = sample_prior(
            self.config.sample.sample_num,
            lambda n: self.manifold.uniform_sample(n).to(self.device),
            self.device,
        )
        x, _, _ = self.sample_backward(init, keep_quiet=True)
        return x, self._to_latlon(x)

    def _jsd_on_hist(self, generated_latlon, reference_latlon):
        hist_ranges = [[-90, 90], [-180, 180]]
        return compute_jsd_2d_histogram(generated_latlon, reference_latlon, bins=30, ranges=hist_ranges)

    def plot_sample(self, samples, savefig=None):
        if isinstance(samples, torch.Tensor): samples = samples.detach().cpu().numpy()
        fig = plt.figure()
        lat, lon = xyz_to_latlon(samples)
        ax = fig.add_subplot(1,1,1, projection=self.projection)

        ax.scatter(lon, lat, s=0.3, color='red', alpha=1.0, label='Samples')
        
        ax.add_feature(cfeature.LAND, zorder = 0, facecolor="#e0e0e0")
        ax.add_feature(cfeature.OCEAN, zorder = 0, facecolor="#b0c4de")
        ax.add_feature(cfeature.COASTLINE, zorder = 1, linewidth=0.5)
        ax.set_global()
        
        ax.set_xlabel('Longitude (degrees)')
        ax.set_ylabel('Latitude (degrees)')
        ax.set_title(f'{samples.shape[0]} Sample plotting on Earth data')
        ax.legend(loc='upper left', fontsize=8, markerscale=0.7)

        plt.savefig(self.savefig_dir + f"/samples_latlon_{savefig}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def validate(self, mode=None, epoch=0, **kwargs):
        if mode == 'start':
            self.best_val_jsd = float("inf")
            self.best_val_epoch = None
            return

        if mode == 'end':
            if self.best_val_epoch is None:
                return

            current_network = self.network
            best_network = load_model(self.best_val_path)
            if best_network is None:
                return

            self.network = best_network.to(self.device)
            x, generated_latlon = self._sample_generated_latlon()
            constraints = get_constraint_metrics(self.manifold, x)
            test_jsd = self._jsd_on_hist(generated_latlon, self.test_latlon)
            full_jsd = self._jsd_on_hist(generated_latlon, self.reference_latlon)
            parts = [
                "test_best",
                f"best_val_epoch={self.best_val_epoch}",
                f"val_JSD on hist.={self.best_val_jsd:.6f}",
                f"test_JSD on hist.={test_jsd:.6f}",
                f"full_JSD on hist.={full_jsd:.6f}",
            ]
            if "mean_eq" in constraints:
                parts.append(f"mean_eq={constraints['mean_eq']:.2e}")
            if "mean_ineq" in constraints:
                parts.append(f"mean_ineq={constraints['mean_ineq']:.2e}")
            logging.info(" | ".join(parts))
            self.network = current_network
            return

        prefix = f"val_epoch_{epoch}"
        x, generated_latlon = self._sample_generated_latlon()
        constraints = get_constraint_metrics(self.manifold, x)
        self.plot_sample(x.cpu().numpy(), savefig=f'sample_epoch_{epoch}')

        val_jsd = self._jsd_on_hist(generated_latlon, self.val_latlon)
        full_jsd = self._jsd_on_hist(generated_latlon, self.reference_latlon)
        log_validation_summary(prefix, self, constraints, [("val_JSD on hist.", val_jsd), ("full_JSD on hist.", full_jsd)])

        if val_jsd < self.best_val_jsd:
            self.best_val_jsd = float(val_jsd)
            self.best_val_epoch = int(epoch)
            save_model(self.validate_dir, self.network, name="model_best_val_jsd.pt")
            logging.info(f"new_best_val | epoch={epoch} | val_JSD on hist.={val_jsd:.6f}")

    def sample_on_manifolds(self):
        logging.info(f'Start sampling on manifolds.')
        if self.network is not None:
            self.network.to(self.device)

        init = sample_prior(
            self.config.sample.sample_num,
            lambda n: self.manifold.uniform_sample(n).to(self.device),
            self.device,
        )
        x, x_hist, _ = self.sample_backward(init, keep_quiet=False)
        log_constraint_metrics(self.manifold, x, prefix="backward_sampling")
        self.plot_sample(x.cpu().numpy(), savefig='generated')
        plot_idx = list(range(0, 100, 10)) + list(range(90, 101))
        for i in range(self.sde.N+1):
            if (100 * i / self.sde.N in plot_idx) or (i > self.sde.N - 5):
                self.plot_sample(x_hist[i].cpu().numpy(), savefig=f'generating_bwd_{i}')

        np.save(f"{self.samples_dir}/{self.dataset_name}_samples_generated.npy", x.cpu().numpy())
        np.save(f"{self.samples_dir}/{self.dataset_name}_samples_test_set.npy", self.test_set.numpy())
        return
