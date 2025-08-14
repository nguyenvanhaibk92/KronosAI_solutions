# utils/phase2_training.py
"""Phase 2 training with pre-trained network correction"""

import jax
import jax.numpy as jnp
from jax import vmap
import optax

from .siren_network import SIREN_neural_one_sample
from .physics_functions import (
    compute_pml_coordinates, complex_mul, epsilon, current_density_Jz
)
from .config import CONFIG, get_source_pos
from .point_generation import generate_fixed_total_points


def net_with_bc_phase2(x, params, params_opt, correction_weight=0.1):
    """
    Network output with boundary conditions for Phase 2 training.
    Combines new trainable parameters with frozen pre-trained parameters.
    
    Args:
        x: Input coordinates
        params: New trainable parameters [params1, params2]
        params_opt: Frozen pre-trained parameters [params1_opt, params2_opt]
        correction_weight: Weight for the correction term (trainable part)
    
    Returns:
        Complex field [real, imag]
    """
    source_pos = get_source_pos()
    x = (x - source_pos)  # Center the input around the source position
    
    params1, params2 = params
    params1_opt, params2_opt = params_opt
    
    # Combine pre-trained (frozen) and correction (trainable) networks
    re = (correction_weight * SIREN_neural_one_sample(x, params1).squeeze() + 
          SIREN_neural_one_sample(x, params1_opt).squeeze())
    
    im = (correction_weight * SIREN_neural_one_sample(x, params2).squeeze() + 
          SIREN_neural_one_sample(x, params2_opt).squeeze())
    
    return jnp.array([re, im])  # Return as complex array [re, im]


def net_grad_phase2(x, params, params_opt, correction_weight=0.1):
    """Compute gradients for Phase 2 network"""
    grad_fn = jax.jacfwd(lambda x_in: net_with_bc_phase2(x_in, params, params_opt, correction_weight), argnums=0)
    return grad_fn(x)


def compute_pml_derivatives_phase2(x, params, params_opt, A, B, correction_weight=0.1):
    """Compute PML derivatives for Phase 2 network"""
    grad_Ez = net_grad_phase2(x, params, params_opt, correction_weight)  # [2, 2] array
    
    # A * ∂Ez/∂x for both real and imaginary parts
    A_dEz_dx = complex_mul(A, grad_Ez[:, 0])  # [:, 0] is ∂/∂x
    
    # B * ∂Ez/∂y for both real and imaginary parts  
    B_dEz_dy = complex_mul(B, grad_Ez[:, 1])  # [:, 1] is ∂/∂y
    
    return A_dEz_dx, B_dEz_dy


def compute_divergence_terms_phase2(x, params, params_opt, A, B, correction_weight=0.1):
    """Compute divergence terms for Phase 2 network"""
    
    # Define functions for second derivatives
    def A_term(pt, A):
        grad_Ez = net_grad_phase2(pt, params, params_opt, correction_weight)
        return complex_mul(A, grad_Ez[:, 0])  # A * ∂Ez/∂x
    
    def B_term(pt, B):
        grad_Ez = net_grad_phase2(pt, params, params_opt, correction_weight)
        return complex_mul(B, grad_Ez[:, 1])  # B * ∂Ez/∂y
    
    # Compute divergence using Jacobian
    jac_A = jax.jacfwd(A_term)(x, A)  # Shape: [2, 2]
    jac_B = jax.jacfwd(B_term)(x, B)  # Shape: [2, 2]
    
    # Extract the relevant derivatives
    div_A_re = jac_A[0, 0]  # ∂/∂x(A*∂Ez_re/∂x) 
    div_A_im = jac_A[1, 0]  # ∂/∂x(A*∂Ez_im/∂x)
    div_B_re = jac_B[0, 1]  # ∂/∂y(B*∂Ez_re/∂y)
    div_B_im = jac_B[1, 1]  # ∂/∂y(B*∂Ez_im/∂y)
    
    # Total divergence term (real and imaginary parts)
    return jnp.array([div_A_re + div_B_re, div_A_im + div_B_im])


@jax.jit
def electromagnetic_pml_residual_single_phase2(params, params_opt, x, correction_weight=0.1):
    """
    Electromagnetic wave equation residual for Phase 2 training
    """
    # Get PML coefficients and field
    A, B, C = compute_pml_coordinates(x)
    Ez = net_with_bc_phase2(x, params, params_opt, correction_weight)
    
    # Compute divergence terms efficiently
    div_term = compute_divergence_terms_phase2(x, params, params_opt, A, B, correction_weight)
    
    # εω² term with C coefficient
    wavenumber = CONFIG.physics.wavenumber
    C_eps_omega2_Ez = complex_mul(C, wavenumber**2 * epsilon(x) * Ez)
    
    # Source term: iω Jz = [0, ω Jz] (purely imaginary)
    i_omega_Jz = jnp.array([1.0, 1.0]) * current_density_Jz(x)
    
    # Complete residual
    residual = div_term + C_eps_omega2_Ez - i_omega_Jz
    
    return residual[0], residual[1]


@jax.jit
def pinn_loss_pml_phase2(params, params_opt, x_collocation, correction_weight=0.1):
    """
    PINN loss for Phase 2 training with correction networks
    """
    # PDE residuals for all points using Phase 2 framework
    residuals_re, residuals_im = vmap(
        lambda x: electromagnetic_pml_residual_single_phase2(params, params_opt, x, correction_weight)
    )(x_collocation)
    squared_residuals = jnp.abs(residuals_re) + jnp.abs(residuals_im)
    
    # Single unified loss
    pde_loss = jnp.mean(squared_residuals)
    
    return pde_loss


@jax.jit
def evaluate_pml_phase2(params, params_opt, x_collocation, correction_weight=0.1):
    """Evaluation function for Phase 2"""
    pde_loss = pinn_loss_pml_phase2(params, params_opt, x_collocation, correction_weight)
    return pde_loss


@jax.jit
def value_and_grad_fn_pml_phase2(params, params_opt, x_collocation, correction_weight=0.1):
    """Value and gradient function for Phase 2"""
    return jax.value_and_grad(
        lambda p: pinn_loss_pml_phase2(p, params_opt, x_collocation, correction_weight)
    )(params)


def train_pml_phase2(params, params_opt, correction_weight=0.1, n_epochs=None, 
                     learning_rate=None, warmup_steps=None, max_grad_norm=None):
    """
    Phase 2 training function using correction networks
    
    Args:
        params: New trainable correction parameters
        params_opt: Frozen pre-trained parameters from Phase 1
        correction_weight: Weight for correction term (default 0.1)
        n_epochs: Number of training epochs (uses config default if None)
        learning_rate: Peak learning rate (uses config default if None)
        warmup_steps: Number of warmup steps (uses config default if None)
        max_grad_norm: Maximum gradient norm for clipping (uses config default if None)
    
    Returns:
        params_corrected: Optimized correction parameters
        history: Training history
    """
    # Use config defaults if not provided and ensure they are numbers
    if n_epochs is None:
        n_epochs = int(CONFIG.training.epochs // 2)  # Use half epochs for Phase 2
    if learning_rate is None:
        learning_rate = float(CONFIG.training.learning_rate) * 0.1  # Lower LR for Phase 2
    if warmup_steps is None:
        warmup_steps = int(CONFIG.training.warmup_steps // 4)  # Shorter warmup
    if max_grad_norm is None:
        max_grad_norm = float(CONFIG.training.max_grad_norm)
    
    history = []
    
    print("\n" + "="*60)
    print("Starting Phase 2 Electromagnetic PINN training (Correction Phase)")
    print("="*60)
    print(f"Correction weight: {correction_weight}")
    print(f"Network architecture: Pre-trained + {correction_weight} × Correction")
    print(f"Phase 2 epochs: {n_epochs}")
    print(f"Phase 2 learning rate: {learning_rate}")
    print(f"Phase 2 warmup steps: {warmup_steps}")
    print()
    
    # Create learning rate schedule with warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=n_epochs - warmup_steps,
        end_value=learning_rate * 0.01
    )
    
    # Adam optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(schedule)
    )
    opt_state = optimizer.init(params)
    
    for t in range(n_epochs):
        # Generate collocation points with larger circle radius for Phase 2
        circle_radius = (CONFIG.points.circle_radius_start * 3 + 
                        t/n_epochs * (CONFIG.domain.xy_in))
        
        x_collocation = generate_fixed_total_points(
            CONFIG.points.base_nx, 
            CONFIG.points.base_ny, 
            CONFIG.points.n_circle_points, 
            circle_radius=circle_radius, 
            random_perturbation=True, 
            key=jax.random.PRNGKey(t + 50000)  # Different seed for Phase 2
        ) 
        
        val, grads = value_and_grad_fn_pml_phase2(params, params_opt, x_collocation, correction_weight)
        
        # Update parameters with clipped gradients
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        if t % 200 == 0:  # More frequent logging for Phase 2
            pde_loss = evaluate_pml_phase2(params, params_opt, x_collocation, correction_weight)
            current_lr = schedule(t)
            grad_norm = optax.global_norm(grads)
            
            print(f"Phase 2 Epoch {t}: Total Loss = {val:.6e}  EM PDE Loss = {pde_loss:.6e}  "
                  f"LR = {current_lr:.2e}  GradNorm = {grad_norm:.4f}")
            history.append((t, float(val), float(pde_loss), float(current_lr), float(grad_norm)))
    
    print(f"\nPhase 2 training completed!")
    print(f"Final Phase 2 loss: {history[-1][1] if history else 'N/A':.6e}")
    
    return params, history


def evaluate_residual_field(params_opt, nx=200):
    """
    Evaluate and visualize the PDE residual field after Phase 1 training
    
    Args:
        params_opt: Optimized parameters from Phase 1
        nx: Grid resolution for evaluation
    
    Returns:
        Residual field arrays for analysis
    """
    from .pinn_model import electromagnetic_pml_residual_single
    
    # Create evaluation grid
    x_plot = jnp.linspace(CONFIG.domain.extended_x[0], CONFIG.domain.extended_x[1], nx)
    y_plot = jnp.linspace(CONFIG.domain.extended_y[0], CONFIG.domain.extended_y[1], nx)
    X, Y = jnp.meshgrid(x_plot, y_plot)
    points = jnp.stack((X.flatten(), Y.flatten()), axis=1)
    
    # Evaluate residuals
    residuals = vmap(lambda x: electromagnetic_pml_residual_single(params_opt, x))(points)
    residual_re = residuals[0].reshape((nx, nx))
    residual_im = residuals[1].reshape((nx, nx))
    residual_magnitude = jnp.sqrt(residual_re**2 + residual_im**2)
    
    return X, Y, residual_re, residual_im, residual_magnitude


def combined_net_with_bc(x, params_phase1, params_phase2, correction_weight=0.1):
    """
    Final combined network using both Phase 1 and Phase 2 parameters
    
    Args:
        x: Input coordinates
        params_phase1: Parameters from Phase 1 training
        params_phase2: Parameters from Phase 2 training (correction)
        correction_weight: Weight for correction term
    
    Returns:
        Combined complex field [real, imag]
    """
    source_pos = get_source_pos()
    x = (x - source_pos)  # Center the input around the source position
    
    params1_p1, params2_p1 = params_phase1
    params1_p2, params2_p2 = params_phase2
    
    # Combine Phase 1 (main) + Phase 2 (correction)
    re = (SIREN_neural_one_sample(x, params1_p1).squeeze() + 
          correction_weight * SIREN_neural_one_sample(x, params1_p2).squeeze())
    
    im = (SIREN_neural_one_sample(x, params2_p1).squeeze() + 
          correction_weight * SIREN_neural_one_sample(x, params2_p2).squeeze())
    
    return jnp.array([re, im])


def run_two_phase_training(initial_params):
    """
    Complete two-phase training pipeline
    
    Args:
        initial_params: Initial network parameters
    
    Returns:
        params_phase1: Optimized parameters from Phase 1
        params_phase2: Optimized correction parameters from Phase 2
        history_phase1: Phase 1 training history
        history_phase2: Phase 2 training history
    """
    from .training import train_pml
    from .siren_network import init_mlp_params
    from jax import random
    
    print("="*60)
    print("TWO-PHASE ELECTROMAGNETIC PINN TRAINING")
    print("="*60)
    
    # Phase 1: Standard training
    print("PHASE 1: Initial Training")
    params_phase1, history_phase1 = train_pml(initial_params)
    
    # Initialize new correction networks for Phase 2
    print("\nInitializing correction networks for Phase 2...")
    rng_phase2 = random.PRNGKey(12345)  # Different seed for Phase 2
    params_correction = [
        init_mlp_params(CONFIG.network.neurons, rng_phase2),
        init_mlp_params(CONFIG.network.neurons, rng_phase2)
    ]
    
    # Phase 2: Correction training
    print("PHASE 2: Correction Training")
    params_phase2, history_phase2 = train_pml_phase2(
        params_correction, 
        params_phase1,
        correction_weight=0.1
    )
    
    return params_phase1, params_phase2, history_phase1, history_phase2