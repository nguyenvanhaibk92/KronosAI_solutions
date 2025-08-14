# utils/point_generation.py
"""Point generation utilities for PINN training"""

import jax
import jax.numpy as jnp
from .config import CONFIG


def generate_fixed_total_points(nx, ny, n_circle_points, circle_radius=0.1, 
                              random_perturbation=False, key=jax.random.PRNGKey(0)):
    """
    Generate uniform mesh + n points from dense circle around source point.
    
    Args:
        nx, ny: Base uniform grid resolution
        n_circle_points: Number of circle points to add to uniform mesh
        circle_radius: Radius of dense circle around source
        random_perturbation: If True, randomly perturb grid points
        key: JAX random key (required if random_perturbation=True)
    
    Returns:
        points: Combined mesh points with exactly nx*ny total points
    """
    
    circle_density = n_circle_points
    
    # 1. Generate uniform background mesh
    x_uniform = jnp.linspace(CONFIG.domain.extended_x[0], CONFIG.domain.extended_x[1], nx)
    y_uniform = jnp.linspace(CONFIG.domain.extended_y[0], CONFIG.domain.extended_y[1], ny)
    X_uniform, Y_uniform = jnp.meshgrid(x_uniform, y_uniform)
    uniform_points = jnp.stack([X_uniform.flatten(), Y_uniform.flatten()], axis=1)
    
    if n_circle_points > 0 and key is not None:
        # 2. Generate dense circle mesh
        circle_nx = circle_density * 2
        circle_ny = circle_density * 2
        
        x_circle = jnp.linspace(CONFIG.physics.source_pos[0] - circle_radius, 
                               CONFIG.physics.source_pos[0] + circle_radius, circle_nx)
        y_circle = jnp.linspace(CONFIG.physics.source_pos[1] - circle_radius, 
                               CONFIG.physics.source_pos[1] + circle_radius, circle_ny)
        X_circle, Y_circle = jnp.meshgrid(x_circle, y_circle)
        circle_points = jnp.stack([X_circle.flatten(), Y_circle.flatten()], axis=1)
        
        # Keep only points inside circle
        circle_distances = jnp.sqrt((circle_points[:, 0] - CONFIG.physics.source_pos[0])**2 + 
                                   (circle_points[:, 1] - CONFIG.physics.source_pos[1])**2)
        inside_circle = circle_points[circle_distances <= circle_radius]
        
        # 3. Pick n_circle_points from circle
        key_select, key_replace = jax.random.split(key)
        n_available = len(inside_circle)
        if n_available >= n_circle_points:
            # Pick n_circle_points from available circle points
            circle_indices = jax.random.choice(key_select, n_available, 
                                             shape=(n_circle_points,), replace=False)
            selected_circle = inside_circle[circle_indices]
        else:
            # Use all available circle points
            selected_circle = inside_circle
            n_circle_points = n_available
        
        # 4. Remove n_circle_points from uniform mesh
        total_uniform = len(uniform_points)
        n_uniform_keep = total_uniform - n_circle_points
        uniform_indices = jax.random.choice(key_replace, total_uniform, 
                                          shape=(n_uniform_keep,), replace=False)
        selected_uniform = uniform_points[uniform_indices]
        
        # 5. Combine meshes
        all_points = jnp.concatenate([selected_uniform, selected_circle], axis=0)
    else:
        # No circle points, just use uniform
        all_points = uniform_points
    
    # 6. Add random perturbation if requested
    if random_perturbation:
        if key is None:
            raise ValueError("Random key required when random_perturbation=True")
        
        # Calculate grid spacing
        dx = (CONFIG.domain.extended_x[1] - CONFIG.domain.extended_x[0]) / (nx - 1)
        dy = (CONFIG.domain.extended_y[1] - CONFIG.domain.extended_y[0]) / (ny - 1)
        
        # Split key for x and y perturbations
        key_x, key_y = jax.random.split(key)
        
        n_total = len(all_points)
        
        # Generate random perturbations
        random_x = jax.random.uniform(key_x, (n_total,), minval=-1.0, maxval=1.0) * dx
        random_y = jax.random.uniform(key_y, (n_total,), minval=-1.0, maxval=1.0) * dy
        
        # Apply perturbations
        perturbations = jnp.stack([random_x, random_y], axis=1)
        all_points = all_points + perturbations
    
    return all_points