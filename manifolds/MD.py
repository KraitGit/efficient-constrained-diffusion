import torch
from torch.func import vmap
from manifolds.general import Manifold_general

def _dihedral(coords):
    """
    Computes the dihedral angle for a single molecular structure.
    Accepts coordinates for 4 atoms.
    """
    r12 = coords[1] - coords[0]
    r23 = coords[2] - coords[1]
    r34 = coords[3] - coords[2]
    
    n1 = torch.cross(r12, r23)
    n2 = torch.cross(r23, r34)
    
    cos_phi = torch.dot(n1, n2)
    sin_phi = torch.dot(n1, r34) * torch.norm(r23)
    
    return torch.atan2(sin_phi, cos_phi)

class Manifold_MD(Manifold_general):
    """
    Manifold class for Alanine Dipeptide, defined as a subclass of Manifold_general.

    This class specifies the dipeptide constraints and relies on the parent class
    for all projection, retraction, and differentiation logic. It can handle
    cases with or without inequality constraints on the psi angle.
    """
    def __init__(self, psi_windows=[(130, 170)], boundary_repulsion=0.1):
        """
        Initializes the manifold by defining dipeptide-specific constraints and
        passing them to the parent Manifold_general class.
        """
        dim = 30
        
        m = 1
        self.phi_target_rad = torch.deg2rad(torch.tensor(-70.0))
        
        if psi_windows:
            self.psi_windows_rad = torch.deg2rad(torch.tensor(psi_windows, dtype=torch.float32))
            l = 1
        else:
            self.psi_windows_rad = torch.empty(0, 2)
            l = 0

        super().__init__(dim=dim, m=m, l=l, h=self._h_phi, g=self._g_psi, boundary_repulsion_rate=boundary_repulsion)

    def angle_phi(self, x):
        """Computes the phi angle for a batch of conformations."""
        atom_indices = [1, 3, 4, 6]
        return vmap(lambda s: _dihedral(s[atom_indices]))(x)

    def angle_psi(self, x):
        """Computes the psi angle for a batch of conformations."""
        atom_indices = [3, 4, 6, 8]
        return vmap(lambda s: _dihedral(s[atom_indices]))(x)

    def _h_phi(self, x):
        """
        Equality constraint for a single sample: h(x) = angle_phi(x) - target = 0
        """
        x = x.reshape(-1, 3)
        phi = _dihedral(x[[1, 3, 4, 6]])
        return (phi - self.phi_target_rad).unsqueeze(0)

    def _g_psi(self, x):
        """
        Inequality constraints for a single sample: g(x) <= 0.
        Returns one value that is negative inside any valid psi window.
        """
        if self.l == 0:
            return torch.empty(0, device=x.device, dtype=x.dtype)
        
        x = x.reshape(-1, 3)
        psi = _dihedral(x[[3, 4, 6, 8]])
        
        lows = self.psi_windows_rad[:, 0].to(psi.device)
        highs = self.psi_windows_rad[:, 1].to(psi.device)
        
        g1 = psi - highs
        g2 = lows - psi
        
        per_window_violations = torch.stack([g1, g2], dim=1)
        distance_to_outside = torch.max(per_window_violations, dim=1).values
        min_distance = torch.min(distance_to_outside)
        return min_distance.unsqueeze(0)

    @torch.enable_grad()
    def constraint_grad_fn(self, samples):
        samples.requires_grad_(True)
        gradients = torch.autograd.grad(
            outputs=self.constraint_fn(samples).sum(),
            inputs=samples,
            create_graph=True,
            retain_graph=True)[0]
        return gradients.detach()
