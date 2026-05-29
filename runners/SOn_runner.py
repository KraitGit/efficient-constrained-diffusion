import logging
import matplotlib.pyplot as plt
import torch
import numpy as np
from runners.Basic_runner import BasicRunner
from src.utils import (
    split_dataset,
    check_memory,
    sample_prior,
    log_constraint_metrics,
    get_constraint_metrics,
    log_task_metric,
    log_validation_summary,
)
from scipy.spatial.distance import jensenshannon


class SOnRunner(BasicRunner):
    def __init__(self, config):
        super().__init__(config)
        self.load_data()

        if not self.is_main_process:
            return

        self.plot_hist(self.test_set_statistics, savefig="true", compare=False)

        uniform = self.manifold.uniform_sample(self.config.sample.sample_num)
        uniform_statistics = self.get_statistics(uniform).numpy()
        self.plot_hist(uniform_statistics, savefig="uniform", compare=False)
        
        if self.config.if_train or self.config.if_sample:
            x_hist = self.training_set_path[:self.config.sample.sample_num].clone().transpose(0,1)

            x = x_hist[-1]
            x = self.filter_sample(x)
            statistics = self.get_statistics(x).cpu().numpy()
            self.plot_hist(statistics, savefig='forward_end')
            plot_idx = list(range(10)) + list(range(10, 101, 10))
            for i in range(self.sde.N+1):
                if (100 * i / self.sde.N in plot_idx) or (i < 5):
                    x_temp = x_hist[i].clone()
                    statistics = self.get_statistics(x_temp).cpu().numpy()
                    self.plot_hist(statistics, savefig=f'generating_fwd_{i}')
            
            statistics_path = self.get_statistics_path(x_hist.cpu().detach()).numpy()
            np.save(f"{self.samples_dir}/{self.dataset_name}_statistics_fwd.npy", statistics_path)

    def load_data(self):
        self.power_list = [1, 2, 4, 5]

        data_ori = torch.tensor(np.load(f"./data/SOn/{self.dataset_name}.npy")).reshape(-1, self.manifold.out_dim)
        self.data_set = data_ori[torch.randperm(data_ori.shape[0])].clone()
        self.training_set, self.test_set, self.val_set = split_dataset(self.data_set, self.config.seed)
        self.reference_statistics = self.get_statistics(self.data_set).cpu().numpy()
        self.test_set_statistics = self.reference_statistics

        if self.config.if_train or self.config.if_sample:
            self.training_set_path, _ = self.generate_path_dataset(self.training_set, keep_quiet=False)
            check_memory(self.training_set_path)

    def filter_sample(self, samples):
        all_mat = samples.reshape(-1, self.manifold.mat_dim, self.manifold.mat_dim).cpu()
        manifolds_idx = torch.where(torch.linalg.det(all_mat) > 0)[0]
        manifolds_idx = manifolds_idx.to(samples.device)
        logging.info(f"The number of samples on the correct connected component: {manifolds_idx.shape[0]}/{samples.shape[0]}, the others are dropped.")
        return samples[manifolds_idx]
    
    def get_statistics(self, samples):
        if not isinstance(samples, torch.Tensor): samples = torch.tensor(samples)

        samples = samples.reshape(-1, self.manifold.mat_dim, self.manifold.mat_dim)
        trace_list = []
        for i in range(4):
            samples_pow = torch.matrix_power(samples, self.power_list[i])
            trace = samples_pow.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
            trace_list.append(trace)
        return torch.cat(trace_list, dim=1)
    
    def get_statistics_path(self, path):
        statistics = torch.zeros(path.shape[0], path.shape[1], 4)
        for i in range(path.shape[0]):
            temp = self.get_statistics(path[i])
            statistics[i] = temp.clone()
        return statistics
    
    def plot_hist(self, statistics, savefig=None, compare=True):
        bins = int(statistics.shape[0] / 100) if statistics.shape[0] > 100 else 10
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        for i, ax in enumerate(axes):
            if compare:
                ax.hist(statistics[:, i], bins=bins, histtype='stepfilled',
                        alpha=0.5, density=True, color='green', label='Generated')
                ax.hist(self.test_set_statistics[:, i], bins=bins, histtype='stepfilled',
                        alpha=0.5, density=True, color='red', label='True')
                if i == 0:
                    ax.legend(fontsize=12)
            else:
                ax.hist(statistics[:, i], bins=bins, alpha=1.0, density=True)

            ax.set_title(rf"$\mathrm{{Tr}}(S^{self.power_list[i]})$", fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=12)

        plt.tight_layout()
        plt.savefig(self.savefig_dir + f"/Hist_Statistics_{savefig}.pdf",
                    bbox_inches="tight")
        plt.close(fig)



    def calculate_jsd_scores(self, generated_stats, reference_stats, bins=100):
        jsd_scores = []
        score_by_name = {}

        for i in range(reference_stats.shape[1]):
            min_val = min(reference_stats[:, i].min(), generated_stats[:, i].min())
            max_val = max(reference_stats[:, i].max(), generated_stats[:, i].max())
            bin_edges = np.linspace(min_val, max_val, bins + 1)

            true_hist, _ = np.histogram(reference_stats[:, i], bins=bin_edges, density=True)
            generated_hist, _ = np.histogram(generated_stats[:, i], bins=bin_edges, density=True)

            epsilon = 1e-10
            true_hist += epsilon
            generated_hist += epsilon

            true_hist /= true_hist.sum()
            generated_hist /= generated_hist.sum()

            jsd = jensenshannon(true_hist, generated_hist)
            jsd_scores.append(jsd)
            score_by_name[f"trace^{self.power_list[i]}"] = jsd

        avg_jsd = np.mean(jsd_scores)
        score_by_name["trace avg"] = avg_jsd
        return score_by_name

    def calculate_and_log_jsd(self, generated_stats, bins=100, prefix="generated"):
        scores = self.calculate_jsd_scores(generated_stats, self.reference_statistics, bins=bins)
        for name, value in scores.items():
            log_task_metric(prefix, f"JSD on {name} hist.", value)
        return scores["trace avg"]

    def validate(self, mode=None, epoch=0, **kwargs):
        if mode == 'start' or mode == 'end':
            return

        prefix = f"val_epoch_{epoch}"
        init = sample_prior(
            self.config.sample.sample_num,
            lambda n: self.manifold.uniform_sample(n).to(self.device),
            self.device,
        )
        x, _, _ = self.sample_backward(init, keep_quiet=True)
        constraints = get_constraint_metrics(self.manifold, x)
        x = self.filter_sample(x)

        statistics_gen = self.get_statistics(x).cpu().numpy()
        self.plot_hist(statistics_gen, savefig=f'val_{epoch}_generated')
        scores = self.calculate_jsd_scores(statistics_gen, self.reference_statistics)
        metrics_by_name = [(f"JSD on {name} hist.", value) for name, value in scores.items()]
        log_validation_summary(prefix, self, constraints, metrics_by_name)

    def sample_on_manifolds(self):
        logging.info(f'Start sampling on manifolds.')
        device = self.device
        if self.network is not None: self.network.to(device)

        logging.info("Start sampling backward SDE.")
        init = sample_prior(
            self.config.sample.sample_num,
            lambda n: self.manifold.uniform_sample(n).to(device),
            device,
        )
        x, x_hist, _ = self.sample_backward(init, keep_quiet=False)
        log_constraint_metrics(self.manifold, x, prefix="backward_sampling")

        self.calculate_constraint(x)
        x = self.filter_sample(x)

        statistics_gen = self.get_statistics(x).cpu().numpy()
        self.plot_hist(statistics_gen, savefig='generated_final')

        self.calculate_and_log_jsd(statistics_gen, prefix="backward_sampling")
        
        plot_idx = list(range(0, 100, 10)) + list(range(90, 101))
        for i in range(self.sde.N+1):
            if (100 * i / self.sde.N in plot_idx) or (i > self.sde.N - 5):
                statistics = self.get_statistics(x_hist[i]).cpu().numpy()
                self.plot_hist(statistics, savefig=f'generating_bwd_{i}')

        statistics_path = self.get_statistics_path(x_hist.cpu().detach()).numpy()
        np.save(f"{self.samples_dir}/{self.dataset_name}_statistics_bwd.npy", statistics_path)

        return
