import torch
import numpy as np
from manifolds.general import Manifold_general

def get_transformation_matrix_batched(alpha, a, d, theta):
    """
    Computes a batch of transformation matrices using the Modified DH convention.
    """
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_alpha = torch.cos(alpha)
    sin_alpha = torch.sin(alpha)
    
    zeros = torch.zeros_like(cos_theta)
    ones = torch.ones_like(cos_theta)

    row1 = torch.stack([cos_theta,              -sin_theta,             zeros,          a                ], dim=1)
    row2 = torch.stack([sin_theta * cos_alpha,  cos_theta * cos_alpha,  -sin_alpha,     -d * sin_alpha   ], dim=1)
    row3 = torch.stack([sin_theta * sin_alpha,  cos_theta * sin_alpha,  cos_alpha,      d * cos_alpha    ], dim=1)
    row4 = torch.stack([zeros,                  zeros,                  zeros,          ones             ], dim=1)

    T = torch.stack([row1, row2, row3, row4], dim=1)
    return T

def forward_kinematics_pytorch_batched(q):
    """
    Computes link positions for a batch of joint configurations
    using the Modified DH convention for the Franka Emika Panda.
    """
    if q.dim() == 1:
        q = q.unsqueeze(0)
        
    batch_size, _ = q.shape
    device = q.device
    dtype = q.dtype
    
    dh_params = torch.tensor([
        [0,        0,       0.333,  0      ],
        [-np.pi/2, 0,       0,      0      ],
        [np.pi/2,  0,       0.316,  0      ],
        [np.pi/2,  0.0825,  0,      0      ],
        [-np.pi/2, -0.0825, 0.384,  0      ],
        [np.pi/2,  0,       0,      0      ],
        [np.pi/2,  0.088,   0,      0      ]
    ], dtype=dtype, device=device)
    
    T_7_ee_params = torch.tensor([0, 0, 0.107, 0], dtype=dtype, device=device)

    T_current = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    link_positions = {0: T_current[:, :3, 3]}

    for i in range(7):
        alpha, a, d, theta_offset = dh_params[i]
        alpha_b, a_b, d_b = alpha.expand(batch_size), a.expand(batch_size), d.expand(batch_size)
        theta = q[:, i] + theta_offset
        
        T_link = get_transformation_matrix_batched(alpha_b, a_b, d_b, theta)
        T_current = torch.bmm(T_current, T_link)
        link_positions[i + 1] = T_current[:, :3, 3]
        
    alpha, a, d, theta_offset = T_7_ee_params
    T_ee = get_transformation_matrix_batched(
        alpha.expand(batch_size), a.expand(batch_size), d.expand(batch_size), 
        torch.full((batch_size,), theta_offset, dtype=dtype, device=device)
    )
    T_final = torch.bmm(T_current, T_ee)
    link_positions[11] = T_final[:, :3, 3]
    return link_positions

class Manifold_Robot(Manifold_general):
    """
    Manifold for the 7-DOF Franka Panda arm with endpoint and obstacle constraints.
    """
    def __init__(self, time_steps=10, target_ee_z=0.205, 
                 obstacles_info=[{'position': [0.4, -0.3, 0.205]}, {'position': [0.4, 0.3, 0.205]}],
                 safety_margin=0.0, obstacle_radius=0.1,
                 start_pos=[0.4, -0.5, 0.205], end_pos=[0.4, 0.5, 0.205],
                 boundary_repulsion_rate=0.1):

        self.input_dim = 14
        self.joint_dim = 7
        dim = self.input_dim * time_steps
        self.time_steps = time_steps

        m = 1
        h_func = self._h_combined_summed
        
        l = 1
        g_func = self._g_combined_summed

        self.target_ee_z = target_ee_z
        self.obstacles_info = obstacles_info 
        self.obstacle_positions = None
        self.safety_margin = safety_margin
        self.obstacle_radius = obstacle_radius
        self.robot_links_for_obstacle_collision = list(range(1, 8))
        self.end_effector_link_index = 11
        
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.start_pos_tensor = None
        self.end_pos_tensor = None

        super().__init__(dim=dim, m=m, l=l, h=h_func, g=g_func, boundary_repulsion_rate=boundary_repulsion_rate)

    def _initialize_tensors(self, device, dtype):
        if self.obstacles_info and (self.obstacle_positions is None or self.obstacle_positions.device != device):
            positions = [info['position'] for info in self.obstacles_info]
            self.obstacle_positions = torch.tensor(positions, device=device, dtype=dtype)
        
        if self.start_pos is not None and (self.start_pos_tensor is None or self.start_pos_tensor.device != device):
            self.start_pos_tensor = torch.tensor(self.start_pos, device=device, dtype=dtype)
        if self.end_pos is not None and (self.end_pos_tensor is None or self.end_pos_tensor.device != device):
            self.end_pos_tensor = torch.tensor(self.end_pos, device=device, dtype=dtype)

    def _convert_cos_sin_to_theta(self, q_cos_sin):
        cos_part = q_cos_sin[..., :self.joint_dim]
        sin_part = q_cos_sin[..., self.joint_dim:]
        q_theta = torch.arctan2(sin_part, cos_part)
        return q_theta
    
    def _h_cos_sin_identity(self, q_flat):
        q_cos_sin = q_flat.reshape(self.time_steps, self.input_dim)
        cos_part = q_cos_sin[..., :self.joint_dim]
        sin_part = q_cos_sin[..., self.joint_dim:]
        identity_violation = cos_part**2 + sin_part**2 - 1.0
        return identity_violation.flatten()

    def _h_combined_summed(self, q_flat):
        """Summed equality violation over height, identity, and endpoints."""
        q_cos_sin = q_flat.reshape(self.time_steps, self.input_dim)
        q_points = self._convert_cos_sin_to_theta(q_cos_sin)
        self._initialize_tensors(q_points.device, q_points.dtype)
        link_positions = forward_kinematics_pytorch_batched(q_points)
        ee_positions = link_positions[self.end_effector_link_index]

        h_z_all = ee_positions[:, 2] - self.target_ee_z
        h_z_sum_sq = torch.sum(h_z_all**2)

        h_identity_violations = self._h_cos_sin_identity(q_flat)
        h_identity_sum_sq = torch.sum(h_identity_violations**2)
        
        start_pos_error_sq = 0.0
        if self.start_pos_tensor is not None:
            actual_start_pos = ee_positions[0]
            start_pos_error_sq = torch.sum((actual_start_pos - self.start_pos_tensor)**2)

        end_pos_error_sq = 0.0
        if self.end_pos_tensor is not None:
            actual_end_pos = ee_positions[-1]
            end_pos_error_sq = torch.sum((actual_end_pos - self.end_pos_tensor)**2)
            
        total_violation = h_z_sum_sq + h_identity_sum_sq + start_pos_error_sq + end_pos_error_sq
        
        return total_violation.unsqueeze(0)

    def _g_obstacle_avoidance(self, q_flat):
        q_cos_sin = q_flat.reshape(self.time_steps, self.input_dim)
        q_points = self._convert_cos_sin_to_theta(q_cos_sin)
        self._initialize_tensors(q_points.device, q_points.dtype)
        
        link_positions = forward_kinematics_pytorch_batched(q_points)
        collision_link_pos = torch.stack([link_positions[idx] for idx in self.robot_links_for_obstacle_collision], dim=1)
        all_distances = torch.linalg.norm(collision_link_pos.unsqueeze(2) - self.obstacle_positions.view(1, 1, -1, 3), dim=3)
        min_dist_to_each_obs = torch.min(all_distances, dim=1).values
        effective_clearance = self.safety_margin + self.obstacle_radius
        g_violations = effective_clearance - min_dist_to_each_obs
        return g_violations.flatten()

    def _g_combined_summed(self, q_flat):
        obs_violation = torch.tensor(0.0, device=q_flat.device, dtype=q_flat.dtype)
        if self.obstacles_info:
            g_obs_all = self._g_obstacle_avoidance(q_flat)
            obs_violation = torch.sum(torch.relu(g_obs_all))
        
        total_violation = obs_violation
        return total_violation.unsqueeze(0)
