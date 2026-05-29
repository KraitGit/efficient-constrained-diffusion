import logging
import torch
import numpy as np
import trimesh
import plotly.graph_objs as go
import plotly.offline as offline

from runners.Basic_runner import BasicRunner
from src.utils import (
    split_dataset,
    check_memory,
    sample_prior,
    log_constraint_metrics,
    get_constraint_metrics,
    log_validation_summary,
)
from src.data_utils import refine_dataset_SDF
from scipy.spatial.distance import jensenshannon



class SDFRunner(BasicRunner):
    def __init__(self, config):
        super().__init__(config)
        self.load_data()

        if not self.is_main_process:
            return

        samples_test = self.training_set[:self.config.sample.sample_num].detach()
        self.plot_sample(samples_test.cpu().numpy(), savefig="training_set")
        self.plot_histogram_on_surface(samples=samples_test.cpu().numpy(), savefig='training_set')

        if self.config.if_train or self.config.if_sample:
            x_hist = self.training_set_path[:self.config.sample.sample_num].clone().transpose(0,1)

            x = x_hist[-1].detach().cpu().numpy()
            self.plot_sample(x, savefig='forward_end')
            self.plot_histogram_on_surface(x, savefig='forward_end')

    def load_data(self):
        data_ori = torch.tensor(np.load(f"./data/{self.obj}/{self.dataset_name}_refined.npy"))

        uniform_sample = self.manifold.uniform_sample(self.config.sample.sample_num).to(self.device)
        self.uniform_sample = refine_dataset_SDF(self.manifold.constraint_fn, uniform_sample)
        self.data_set = data_ori[torch.randperm(data_ori.shape[0])].clone()
        self.training_set, self.test_set, self.val_set = split_dataset(self.data_set, self.config.seed)
        self.mesh = self.manifold.mesh

        if self.config.if_train or self.config.if_sample:
            self.training_set_path, _ = self.generate_path_dataset(self.training_set, keep_quiet=False)
            check_memory(self.training_set_path)

        self.reference_histogram = self.compute_histogram_on_surface(self.data_set.cpu().numpy())
        if self.obj == "bunny":
            self.scene_dict = dict(xaxis=dict(range=(-1.05, 1.05), autorange=False),
                            yaxis=dict(range=(-1.05, 1.05), autorange=False),
                            zaxis=dict(range=(-1.05, 1.05), autorange=False),
                            aspectratio=dict(x=1, y=1, z=1),
                            camera=dict(
                            eye=dict(x=-0.5, y=0, z=-2),
                            up=dict(x=0, y=1, z=0),
                            center=dict(x=0, y=0, z=0)))
        else:
            self.scene_dict = dict(xaxis=dict(range=(-1.05, 1.05), autorange=False),
                            yaxis=dict(range=(-1.05, 1.05), autorange=False),
                            zaxis=dict(range=(-1.05, 1.05), autorange=False),
                            aspectratio=dict(x=1, y=1, z=1),
                            camera=dict(
                            eye=dict(x=-1, y=1, z=1),
                            up=dict(x=0, y=1, z=0),
                            center=dict(x=0, y=0, z=0)))

    def plot_sample(self, samples, savefig=None):
        verts = self.mesh.vertices
        I, J, K = self.mesh.faces.transpose()

        trace = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, alphahull=5, opacity=0.4, color='cyan'),
                 go.Scatter3d(x=samples[:, 0], y=samples[:, 1], z=samples[:, 2], mode='markers', marker=dict(size=3))]
        fig = go.Figure(data=trace)
        fig.update_layout(title=f'{samples.shape[0]} scatters',
                          scene=self.scene_dict, width=1400, height=1400, showlegend=True)

        filename0 = self.savefig_dir + f"/Samples_{savefig}"
        offline.plot(fig, filename=f'{filename0}.html', auto_open=False)
        fig.write_image(f"{filename0}.png")

    def compute_histogram_on_surface(self, samples):
        """
        Computes the probability distribution (histogram) of samples over the mesh faces.
        """
        _, _, closest_faces = trimesh.proximity.closest_point(self.mesh, samples)
        unique_faces, counts = np.unique(closest_faces, return_counts=True)
        probs = np.zeros(len(self.mesh.faces))
        probs[unique_faces] = counts / len(samples)
        epsilon = 1e-12
        return probs + epsilon

    def plot_histogram_on_surface(self, samples, colorscale=None, savefig=None):

        verts = self.mesh.vertices
        I, J, K = self.mesh.faces.transpose()

        closest_points, _, closest_faces = trimesh.proximity.closest_point(self.mesh, samples)
        unique_faces, counts = np.unique(closest_faces, return_counts=True)
        probs = np.zeros(len(self.mesh.faces))
        probs[unique_faces] = counts / len(samples)
        densities = probs / self.mesh.area_faces
        densities[np.isnan(densities)] = 0

        cmin, cmax = -0.1, np.percentile(densities, 95) if colorscale is None else colorscale

        traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name='Samples_hist',
                            opacity=1.0, intensity=densities, intensitymode="cell", colorscale="Viridis",
                            cmin=cmin, cmax=cmax)]
        layout = go.Layout(title=f'Histgram of {samples.shape[0]} scatters', scene=self.scene_dict, width=1400, height=1400, showlegend=True)
        fig = go.Figure(data=traces, layout=layout)

        filename0 = self.savefig_dir + f"/Histgram_{savefig}"
        offline.plot(fig, filename=f'{filename0}.html', auto_open=False)
        fig.write_image(f"{filename0}.png")

    def validate(self, mode=None, epoch=0, **kwargs):
        if mode == 'start' or mode == 'end':
            return

        prefix = f"val_epoch_{epoch}"

        init = sample_prior(
            self.config.sample.sample_num,
            lambda n: self.uniform_sample[:n].to(self.device),
            self.device,
        )
        x, _, _ = self.sample_backward(init, keep_quiet=True)
        constraints = get_constraint_metrics(self.manifold, x)

        x = x.cpu().numpy()
        self.plot_sample(x, savefig=f'val_{epoch}_generated')
        self.plot_histogram_on_surface(x, savefig=f'val_{epoch}_generated')

        probs_generated = self.compute_histogram_on_surface(x)
        jsd = jensenshannon(probs_generated, self.reference_histogram) ** 2
        log_validation_summary(prefix, self, constraints, [("JSD on hist.", jsd)])

    def sample_on_manifolds(self):
        logging.info(f'Start sampling on manifolds.')
        device = self.device
        if self.network is not None: self.network.to(device)
        self.manifold.model.to(device)

        logging.info("Start sampling backward SDE.")
        init = sample_prior(
            self.config.sample.sample_num,
            lambda n: self.uniform_sample[:n].to(device),
            device,
        )
        x, x_hist, _ = self.sample_backward(init, keep_quiet=False)
        log_constraint_metrics(self.manifold, x, prefix="backward_sampling")
        self.calculate_constraint(x)
        x = x.cpu().numpy()
        self.plot_sample(x, savefig='generated')
        self.plot_histogram_on_surface(x, savefig='generated')
        plot_idx = list(range(0, 100, 10)) + list(range(90, 101))
        for i in range(self.sde.N+1):
            if (100 * i / self.sde.N in plot_idx) or (i > self.sde.N - 5):
                x_temp = x_hist[i].cpu().numpy()
                self.plot_sample(x_temp, savefig=f'generating_bwd_{i}')
                self.plot_histogram_on_surface(x_temp, savefig=f'generating_bwd_{i}')

        np.save(f"{self.samples_dir}/{self.dataset_name}_samples_generated.npy", x)
        
        return
