# utils/plots.py
"""Visualization functions for electromagnetic PINN simulation"""

import jax
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt

from .config import CONFIG, get_source_pos
from .point_generation import generate_fixed_total_points
from .physics_functions import epsilon
from .pinn_model import net_with_bc
from .phase2_training import net_with_bc_phase2
from .utils import compute_field_magnitude


def visualize_permittivity(nx=200):
    """Visualize the permittivity function"""
    x_plot = jnp.linspace(CONFIG.domain.extended_x[0], CONFIG.domain.extended_x[1], nx)
    y_plot = jnp.linspace(CONFIG.domain.extended_y[0], CONFIG.domain.extended_y[1], nx)
    X, Y = jnp.meshgrid(x_plot, y_plot)
    points = jnp.stack((X.flatten(), Y.flatten()), axis=1)
    q = vmap(epsilon)(points)
    
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, q.reshape((nx, nx)), cmap='RdBu', shading='auto')
    plt.colorbar(label='Permittivity')
    plt.title('Permittivity Distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Mark interior domain boundaries
    interior_x = CONFIG.domain.interior_x
    interior_y = CONFIG.domain.interior_y
    plt.axvline(x=interior_x[0], color='black', linestyle='--', alpha=0.7, label='Interior boundary')
    plt.axvline(x=interior_x[1], color='black', linestyle='--', alpha=0.7)
    plt.axhline(y=interior_y[0], color='black', linestyle='--', alpha=0.7)
    plt.axhline(y=interior_y[1], color='black', linestyle='--', alpha=0.7)
    
    # Mark source position
    source_pos = get_source_pos()
    plt.scatter(source_pos[0], source_pos[1], 
                c='red', s=100, marker='*', label='Source', zorder=5)
    
    plt.legend(loc='upper right')
    plt.show()


def visualize_points_distribution():
    """Visualize the point distribution used for training"""
    pts = generate_fixed_total_points(
        CONFIG.points.base_nx, 
        CONFIG.points.base_ny, 
        CONFIG.points.n_circle_points, 
        circle_radius=CONFIG.points.circle_radius_start, 
        random_perturbation=True, 
        key=jax.random.PRNGKey(3)
    )
    
    plt.figure(figsize=(8, 7))
    plt.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.5)
    plt.title('Training Point Distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Mark domain boundaries
    interior_x = CONFIG.domain.interior_x
    interior_y = CONFIG.domain.interior_y
    plt.axvline(x=interior_x[0], color='red', linestyle='--', alpha=0.7, label='Interior boundary')
    plt.axvline(x=interior_x[1], color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=interior_y[0], color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=interior_y[1], color='red', linestyle='--', alpha=0.7)
    
    # Mark source position
    source_pos = get_source_pos()
    plt.scatter(source_pos[0], source_pos[1], 
                c='red', s=100, marker='*', label='Source', zorder=5)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    return pts


def visualize_solution(params, nx=200):
    """Visualize the trained electromagnetic field solution"""
    x_plot = jnp.linspace(CONFIG.domain.extended_x[0], CONFIG.domain.extended_x[1], nx)
    y_plot = jnp.linspace(CONFIG.domain.extended_y[0], CONFIG.domain.extended_y[1], nx)
    X, Y = jnp.meshgrid(x_plot, y_plot)
    points = jnp.stack((X.flatten(), Y.flatten()), axis=1)
    q = vmap(epsilon)(points)
    
    # Get network predictions
    Ez_complex = vmap(lambda x: net_with_bc(x, params))(points)
    Ez_real = Ez_complex[:, 0].reshape((nx, nx))
    Ez_imag = Ez_complex[:, 1].reshape((nx, nx))
    Ez_magnitude = compute_field_magnitude(Ez_complex).reshape((nx, nx))

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    source_pos = get_source_pos()

    # Real part
    im1 = axes[0].pcolormesh(X, Y, Ez_real, cmap='bwr', shading='auto', vmin=-0.05, vmax=0.05)
    axes[0].scatter(X.flatten()[(q + jnp.roll(q,1))==3], Y.flatten()[(q + jnp.roll(q,1))==3], c='k', s=1, marker='.', alpha=0.1)
    axes[0].set_title('Real(Ez)')
    axes[0].scatter(source_pos[0], source_pos[1], c='black', s=50, marker='*')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].pcolormesh(X, Y, Ez_imag, cmap='bwr', shading='auto', vmin=-0.05, vmax=0.05)
    axes[1].scatter(X.flatten()[(q + jnp.roll(q,1))==3], Y.flatten()[(q + jnp.roll(q,1))==3], c='k', s=1, marker='.', alpha=0.1)
    axes[1].set_title('Imag(Ez)')
    axes[1].scatter(source_pos[0], source_pos[1], c='black', s=50, marker='*')
    plt.colorbar(im2, ax=axes[1])

    # Add domain boundaries to all plots
    for ax in axes:
        interior_x = CONFIG.domain.interior_x
        interior_y = CONFIG.domain.interior_y
        ax.axvline(x=interior_x[0], color='white', linestyle='--', alpha=0.8, linewidth=2)
        ax.axvline(x=interior_x[1], color='white', linestyle='--', alpha=0.8, linewidth=2)
        ax.axhline(y=interior_y[0], color='white', linestyle='--', alpha=0.8, linewidth=2)
        ax.axhline(y=interior_y[1], color='white', linestyle='--', alpha=0.8, linewidth=2)

    plt.tight_layout()
    plt.show()


def visualize_solution_phase2(params, params_opt, nx=200):
    """Visualize the trained electromagnetic field solution"""
    x_plot = jnp.linspace(CONFIG.domain.extended_x[0], CONFIG.domain.extended_x[1], nx)
    y_plot = jnp.linspace(CONFIG.domain.extended_y[0], CONFIG.domain.extended_y[1], nx)
    X, Y = jnp.meshgrid(x_plot, y_plot)
    points = jnp.stack((X.flatten(), Y.flatten()), axis=1)
    q = vmap(epsilon)(points)
    
    # Get network predictions
    Ez_complex = vmap(lambda x: net_with_bc_phase2(x, params, params_opt))(points)
    Ez_real = Ez_complex[:, 0].reshape((nx, nx))
    Ez_imag = Ez_complex[:, 1].reshape((nx, nx))
    Ez_magnitude = compute_field_magnitude(Ez_complex).reshape((nx, nx))

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    source_pos = get_source_pos()

    # Real part
    im1 = axes[0].pcolormesh(X, Y, Ez_real, cmap='bwr', shading='auto', vmin=-0.05, vmax=0.05)
    axes[0].scatter(X.flatten()[(q + jnp.roll(q,1))==3], Y.flatten()[(q + jnp.roll(q,1))==3], c='k', s=1, marker='.', alpha=0.1)
    axes[0].set_title('Real(Ez)')
    axes[0].scatter(source_pos[0], source_pos[1], c='black', s=50, marker='*')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].pcolormesh(X, Y, Ez_imag, cmap='bwr', shading='auto', vmin=-0.05, vmax=0.05)
    axes[1].scatter(X.flatten()[(q + jnp.roll(q,1))==3], Y.flatten()[(q + jnp.roll(q,1))==3], c='k', s=1, marker='.', alpha=0.1)
    axes[1].set_title('Imag(Ez)')
    axes[1].scatter(source_pos[0], source_pos[1], c='black', s=50, marker='*')
    plt.colorbar(im2, ax=axes[1])

    # Add domain boundaries to all plots
    for ax in axes:
        interior_x = CONFIG.domain.interior_x
        interior_y = CONFIG.domain.interior_y
        ax.axvline(x=interior_x[0], color='white', linestyle='--', alpha=0.8, linewidth=2)
        ax.axvline(x=interior_x[1], color='white', linestyle='--', alpha=0.8, linewidth=2)
        ax.axhline(y=interior_y[0], color='white', linestyle='--', alpha=0.8, linewidth=2)
        ax.axhline(y=interior_y[1], color='white', linestyle='--', alpha=0.8, linewidth=2)

    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """Plot training history"""
    epochs = [h[0] for h in history]
    total_loss = [h[1] for h in history]
    pde_loss = [h[2] for h in history]
    learning_rate = [h[3] for h in history]
    grad_norm = [h[4] for h in history]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss plots
    axes[0, 0].semilogy(epochs, total_loss, label='Total Loss', linewidth=2)
    axes[0, 0].semilogy(epochs, pde_loss, label='PDE Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[0, 1].semilogy(epochs, learning_rate, color='orange', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].set_title('Learning Rate Schedule')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gradient norm
    axes[1, 0].plot(epochs, grad_norm, color='green', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].set_title('Gradient Norm')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined loss plot
    axes[1, 1].loglog(epochs, total_loss, label='Total Loss', linewidth=2)
    axes[1, 1].loglog(epochs, pde_loss, label='PDE Loss', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Log-Log Loss Plot')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_field_components_separately(params, nx=200):
    """Create separate figures for each field component"""
    x_plot = jnp.linspace(CONFIG.domain.extended_x[0], CONFIG.domain.extended_x[1], nx)
    y_plot = jnp.linspace(CONFIG.domain.extended_y[0], CONFIG.domain.extended_y[1], nx)
    X, Y = jnp.meshgrid(x_plot, y_plot)
    points = jnp.stack((X.flatten(), Y.flatten()), axis=1)
    
    # Get network predictions
    Ez_complex = vmap(lambda x: net_with_bc(x, params))(points)
    Ez_real = Ez_complex[:, 0].reshape((nx, nx))
    Ez_imag = Ez_complex[:, 1].reshape((nx, nx))
    Ez_magnitude = compute_field_magnitude(Ez_complex).reshape((nx, nx))
    source_pos = get_source_pos()
    
    # Real part
    plt.figure(figsize=(8, 6))
    im1 = plt.pcolormesh(X, Y, Ez_real, cmap='RdBu', shading='auto')
    plt.title('Real(Ez)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(source_pos[0], source_pos[1], c='black', s=50, marker='*', label='Source')
    plt.colorbar(im1, label='Real(Ez)')
    plt.legend()
    plt.show()
    
    # Imaginary part
    plt.figure(figsize=(8, 6))
    im2 = plt.pcolormesh(X, Y, Ez_imag, cmap='RdBu', shading='auto')
    plt.title('Imag(Ez)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(source_pos[0], source_pos[1], c='black', s=50, marker='*', label='Source')
    plt.colorbar(im2, label='Imag(Ez)')
    plt.legend()
    plt.show()
    
    # Magnitude
    plt.figure(figsize=(8, 6))
    im3 = plt.pcolormesh(X, Y, Ez_magnitude, cmap='viridis', shading='auto')
    plt.title('|Ez|')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(source_pos[0], source_pos[1], c='red', s=50, marker='*', label='Source')
    plt.colorbar(im3, label='|Ez|')
    plt.legend()
    plt.show()


def save_field_plots(params, filename_prefix="field_plot", nx=200, dpi=300):
    """Save field plots to files"""
    x_plot = jnp.linspace(CONFIG.domain.extended_x[0], CONFIG.domain.extended_x[1], nx)
    y_plot = jnp.linspace(CONFIG.domain.extended_y[0], CONFIG.domain.extended_y[1], nx)
    X, Y = jnp.meshgrid(x_plot, y_plot)
    points = jnp.stack((X.flatten(), Y.flatten()), axis=1)
    
    # Get network predictions
    Ez_complex = vmap(lambda x: net_with_bc(x, params))(points)
    Ez_real = Ez_complex[:, 0].reshape((nx, nx))
    Ez_imag = Ez_complex[:, 1].reshape((nx, nx))
    Ez_magnitude = compute_field_magnitude(Ez_complex).reshape((nx, nx))
    source_pos = get_source_pos()
    
    # Combined plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Real part
    im1 = axes[0].pcolormesh(X, Y, Ez_real, cmap='RdBu', shading='auto')
    axes[0].set_title('Real(Ez)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].scatter(source_pos[0], source_pos[1], c='black', s=50, marker='*', label='Source')
    plt.colorbar(im1, ax=axes[0])
    axes[0].legend()
    
    # Imaginary part
    im2 = axes[1].pcolormesh(X, Y, Ez_imag, cmap='RdBu', shading='auto')
    axes[1].set_title('Imag(Ez)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].scatter(source_pos[0], source_pos[1], c='black', s=50, marker='*', label='Source')
    plt.colorbar(im2, ax=axes[1])
    axes[1].legend()
    
    # Magnitude
    im3 = axes[2].pcolormesh(X, Y, Ez_magnitude, cmap='viridis', shading='auto')
    axes[2].set_title('|Ez|')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].scatter(source_pos[0], source_pos[1], c='red', s=50, marker='*', label='Source')
    plt.colorbar(im3, ax=axes[2])
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_combined.png", dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Field plots saved as {filename_prefix}_combined.png")