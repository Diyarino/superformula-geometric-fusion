# -*- coding: utf-8 -*-
"""
Superformula Fusion Layer Module.

This module implements a differentiable fusion layer based on the Gielis Superformula.
It projects multimodal input data (e.g., visual and sensor features) onto a learned
geometric manifold ('supershape'), allowing the network to dynamically adapt the
topology of the latent space to the data structure.

Created on: Mon Feb  9 07:44:03 2026
Author: Altinses
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class SuperformulaFusion(nn.Module):
    """
    A neural fusion layer that projects two input modalities onto the surface
    of a geometric 'supershape' defined by the Gielis Superformula.

    The parameters of the shape (m, n1, n2, n3) are learnable, allowing the
    manifold's geometry to adapt during training. This creates a structured
    latent space where the relationship between modalities is encoded geometrically.

    Attributes:
        input_dim (int): The dimensionality of the input feature vectors.
        proj_phi (nn.Linear): Linear projection for Modality A to latitudinal angle.
        proj_theta (nn.Linear): Linear projection for Modality B to longitudinal angle.
        norm_a (nn.LayerNorm): Layer normalization for Modality A.
        norm_b (nn.LayerNorm): Layer normalization for Modality B.
        scale_phi (nn.Parameter): Learnable scaling factor for the Phi angle.
        scale_theta (nn.Parameter): Learnable scaling factor for the Theta angle.
        raw_m (nn.Parameter): Unconstrained parameter for rotational symmetry (m).
        raw_n1 (nn.Parameter): Unconstrained parameter for shape convexity (n1).
        raw_n2 (nn.Parameter): Unconstrained parameter for shape convexity (n2).
        raw_n3 (nn.Parameter): Unconstrained parameter for shape convexity (n3).
        decoder_a_scale (nn.Parameter): Scale factors for reconstructing Modality A.
        decoder_b_scale (nn.Parameter): Scale factors for reconstructing Modality B.
        points (torch.Tensor): Stores the most recent 3D projection points.
        params (torch.Tensor): Stores the most recent shape parameters.
    """

    def __init__(self, input_dim: int = 288):
        """
        Initializes the SuperformulaFusion layer.

        Args:
            input_dim (int): The dimension of the input feature vectors for both modalities.
                             Defaults to 288.
        """
        super().__init__()
        self.input_dim = input_dim

        # --- A. Input Projection ---
        # Projects high-dimensional inputs to scalar angles (Phi and Theta)
        self.proj_phi = nn.Linear(input_dim, 1)    # Modality A -> Latitude (Phi)
        self.proj_theta = nn.Linear(input_dim, 1)  # Modality B -> Longitude (Theta)

        self.norm_a = nn.LayerNorm(input_dim)
        self.norm_b = nn.LayerNorm(input_dim)

        # Learnable scaling factors ("Temperature") to control the angular spread
        self.scale_phi = nn.Parameter(torch.tensor(1.0))
        self.scale_theta = nn.Parameter(torch.tensor(1.0))

        # --- B. Learnable Shape Parameters ---
        # Initialized to approximate a sphere (m=0, n=2).
        # We store 'raw' parameters to allow unconstrained optimization,
        # usually mapped to valid ranges via activation functions later.
        self.raw_m = nn.Parameter(torch.tensor(0.0))
        # Initialize n parameters to log(2.0) so that exp(raw_n) approx 2.0
        self.raw_n1 = nn.Parameter(torch.tensor(np.log(2.0)))
        self.raw_n2 = nn.Parameter(torch.tensor(np.log(2.0)))
        self.raw_n3 = nn.Parameter(torch.tensor(np.log(2.0)))

        # Simple scaling weights for a pseudo-decoder functionality
        self.decoder_a_scale = nn.Parameter(torch.ones(input_dim))
        self.decoder_b_scale = nn.Parameter(torch.ones(input_dim))

        # Internal state storage for visualization
        self.points = None
        self.params = None

    def get_shape_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves the valid Gielis parameters from the internal raw parameters.

        Applies constraints to ensure numerical stability and valid shapes:
        - m: Softplus ensures positivity.
        - n1, n2, n3: Exp ensures positivity, +0.1 prevents division by zero.

        Returns:
            Tuple[torch.Tensor, ...]: A tuple containing (m, n1, n2, n3).
        """
        m = F.softplus(self.raw_m)
        n1 = torch.exp(self.raw_n1) + 0.1
        n2 = torch.exp(self.raw_n2) + 0.1
        n3 = torch.exp(self.raw_n3) + 0.1
        return m, n1, n2, n3

    def superformula_radius(self, angle: torch.Tensor, m: float, n1: float,
                            n2: float, n3: float, a: float = 1.0, b: float = 1.0) -> torch.Tensor:
        """
        Computes the radius 'r' for a given angle using the Gielis Superformula.

        Formula:
            r(\phi) = [ |cos(m\phi / 4) / a|^n2 + |sin(m\phi / 4) / b|^n3 ] ^ (-1/n1)

        Args:
            angle (torch.Tensor): The input angle (phi or theta).
            m (float): Symmetry parameter.
            n1, n2, n3 (float): Form/curvature parameters.
            a, b (float, optional): Scaling factors. Defaults to 1.0.

        Returns:
            torch.Tensor: The computed radius at the given angle.
        """
        part1 = torch.abs(torch.cos(m * angle / 4.0) / a) ** n2
        part2 = torch.abs(torch.sin(m * angle / 4.0) / b) ** n3
        # Add epsilon for numerical stability before power operation
        sum_parts = part1 + part2 + 1e-6
        return sum_parts ** (-1.0 / n1)

    def spherical_to_cartesian(self, phi: torch.Tensor, theta: torch.Tensor,
                               m: float, n1: float, n2: float, n3: float) -> Tuple[torch.Tensor, ...]:
        """
        Converts spherical coordinates (angles) to 3D Cartesian coordinates (x, y, z)
        based on the superformula radius.

        Args:
            phi (torch.Tensor): Latitude angle [-pi/2, pi/2].
            theta (torch.Tensor): Longitude angle [-pi, pi].
            m, n1, n2, n3 (float): Shape parameters.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The (x, y, z) coordinates.
        """
        r1 = self.superformula_radius(theta, m, n1, n2, n3)
        r2 = self.superformula_radius(phi, m, n1, n2, n3)
        r = r1 * r2

        x = r * torch.cos(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.cos(phi)
        z = r * torch.sin(phi)

        return x, y, z

    def cartesian_to_spherical(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts 3D Cartesian coordinates back to spherical angles.

        Args:
            x, y, z (torch.Tensor): Input Cartesian coordinates.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The reconstructed angles (phi, theta).
        """
        # 1. Compute Radius
        r = torch.sqrt(x**2 + y**2 + z**2 + 1e-6)

        # 2. Compute Theta (Longitude)
        # atan2 returns values in range [-pi, pi]
        theta = torch.atan2(y, x)

        # 3. Compute Phi (Latitude)
        # z = r * sin(phi)  =>  phi = arcsin(z / r)
        # Clamp input to asin to avoid NaNs due to numerical precision errors > 1.0
        phi = torch.asin(torch.clamp(z / r, -0.99, 0.99))

        return phi, theta

    def forward(self, mod_a: torch.Tensor, mod_b: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the fusion layer.

        Maps high-dimensional modality vectors to a 3D point on the learnable surface.

        Args:
            mod_a (torch.Tensor): Input features for Modality A (Batch, Input_Dim).
            mod_b (torch.Tensor): Input features for Modality B (Batch, Input_Dim).

        Returns:
            torch.Tensor: The fused 3D coordinates (Batch, 3).
        """
        # 1. Normalize Inputs
        a_norm = self.norm_a(mod_a)
        b_norm = self.norm_b(mod_b)

        # 2. Retrieve current shape parameters
        m, n1, n2, n3 = self.get_shape_params()

        # 3. Map to Angular Space
        # Use tanh to bound the output, then scale to appropriate angular ranges.
        # Phi (Latitude): range approx [-pi/2, pi/2]
        # Theta (Longitude): range approx [-pi, pi]
        phi = torch.tanh(a_norm @ self.proj_phi.weight.T) * (np.pi / 2.0) * 0.95 * self.scale_phi
        theta = torch.tanh(b_norm @ self.proj_theta.weight.T) * np.pi * self.scale_theta

        # 4. Map to Cartesian Space (The "Fusion")
        x, y, z = self.spherical_to_cartesian(phi, theta, m, n1, n2, n3)

        # Store state for potential visualization/loss calculation
        self.points = torch.stack([x, y, z], dim=2)
        self.params = torch.stack((m, n1, n2, n3))

        return self.points

    def decode(self, fused_xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstructs the original modality features from the fused 3D points.

        Args:
            fused_xyz (torch.Tensor): The 3D coordinates (Batch, 3).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Reconstructed features for Modality A and B.
        """
        x = fused_xyz[:, :, 0]
        y = fused_xyz[:, :, 1]
        z = fused_xyz[:, :, 2]

        phi, theta = self.cartesian_to_spherical(x, y, z)

        # Simple linear reconstruction: Angle * Scale_Vector
        recon_mod_a = phi * self.decoder_a_scale
        recon_mod_b = theta * self.decoder_b_scale

        return recon_mod_a, recon_mod_b

    def visualize(self, idx: int = 0, ax: Optional[plt.Axes] = None) -> None:
        """
        Visualizes the learned geometric manifold and the projected data points.

        Generates a 3D wireframe of the current supershape and scatters the
        projected points from the batch onto it.

        Args:
            idx (int, optional): The batch index to visualize (if tracking history). Defaults to 0.
            ax (matplotlib.axes.Axes, optional): An existing 3D axes object.
                                                 If None, a new figure is created.
        """
        self.eval() # Switch to evaluation mode

        # Detach tensors for numpy conversion
        m_ten, n1_ten, n2_ten, n3_ten = self.params.detach()
        points_np = self.points.detach().cpu().numpy()

        # Handle batch dimension if points are (Batch, 3) or (Batch, 1, 3)
        if points_np.ndim == 3:
            points_np = points_np.reshape(-1, 3)

        m = m_ten.item()
        n1, n2, n3 = n1_ten.item(), n2_ten.item(), n3_ten.item()

        # Generate Grid for the Surface Mesh
        res = 60
        theta_grid = torch.linspace(-np.pi, np.pi, res)
        phi_grid = torch.linspace(-np.pi / 2, np.pi / 2, res)
        THETA, PHI = torch.meshgrid(theta_grid, phi_grid, indexing='ij')

        r1 = self.superformula_radius(THETA, m, n1, n2, n3)
        r2 = self.superformula_radius(PHI, m, n1, n2, n3)
        R = r1 * r2

        X_grid = R * torch.cos(THETA) * torch.cos(PHI)
        Y_grid = R * torch.sin(THETA) * torch.cos(PHI)
        Z_grid = R * torch.sin(PHI)

        # --- NORMALIZATION FOR VISUALIZATION ---
        # Normalize both grid and points by the global maximum to fit in unit cube [-1, 1]
        max_grid = torch.max(torch.abs(torch.stack([X_grid, Y_grid, Z_grid])))
        max_points = np.max(np.abs(points_np))
        scale_factor = max(max_grid.item(), max_points) + 1e-6

        X_grid /= scale_factor
        Y_grid /= scale_factor
        Z_grid /= scale_factor
        points_norm = points_np / scale_factor

        # --- PLOTTING ---
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='3d')
            external_plot = False
        else:
            external_plot = True

        # Plot 1: The Learned Manifold (Wireframe/Surface)
        # Using a transparent surface with edges to visualize the geometry
        ax.plot_surface(
            X_grid.numpy(), Y_grid.numpy(), Z_grid.numpy(),
            cmap='viridis', alpha=0.3, edgecolor='k', linewidth=0.1
        )

        # Plot 2: The Projected Data Points
        ax.scatter(
            points_norm[:, 0], points_norm[:, 1], points_norm[:, 2],
            c=points_norm[:, 2], cmap='plasma', s=20, alpha=0.8, label='Latent Projections'
        )

        # Labels and Styling
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        # Enforce equal aspect ratio for correct geometric interpretation
        ax.set_box_aspect([1, 1, 1])

        if not external_plot:
            plt.tight_layout()
            plt.show()

        self.train() # Revert to training mode

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print("Initializing SuperformulaFusion Layer...")

    # 1. Setup Layer
    input_dim = 288
    batch_size = 32
    fusion_layer = SuperformulaFusion(input_dim=input_dim)

    # 2. Modify initial parameters to show a distinct shape (e.g., Star shape)
    with torch.no_grad():
        # Setting m to approx 4.5 creates a shape with symmetries
        fusion_layer.raw_m.fill_(4.5)

    # 3. Create Dummy Data
    print(f"Generating random input data (Batch: {batch_size}, Dim: {input_dim})...")
    data_a = torch.randn(batch_size, input_dim)
    data_b = torch.randn(batch_size, input_dim)

    # 4. Forward Pass
    print("Performing Forward Pass...")
    fused_embedding = fusion_layer(data_a, data_b)
    print(f"Fused Output Shape: {fused_embedding.shape}")

    # 5. Decode Pass (Reconstruction)
    print("Performing Reconstruction...")
    rec_a, rec_b = fusion_layer.decode(fused_embedding)
    print(f"Reconstructed Shape A: {rec_a.shape}")

    # 6. Visualization
    print("Visualizing learned manifold...")
    fusion_layer.visualize()
    print("Done.")