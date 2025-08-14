# utils/training.py
"""Training functions for PINN model"""

import jax
import optax
from .pinn_model import value_and_grad_fn_pml, evaluate_pml
from .point_generation import generate_fixed_total_points
from .config import CONFIG


def train_pml(params, n_epochs=None, learning_rate=None, 
              warmup_steps=None, max_grad_norm=None):
    """
    Training function using electromagnetic wave equation with PML
    Enhanced with warmup and gradient clipping
    
    Args:
        params: Initial network parameters
        n_epochs: Number of training epochs (uses config default if None)
        learning_rate: Peak learning rate (uses config default if None)
        warmup_steps: Number of warmup steps (uses config default if None)
        max_grad_norm: Maximum gradient norm for clipping (uses config default if None)
    
    Returns:
        params_opt: Optimized parameters
        history: Training history
    """
    # Use config defaults if not provided
    if n_epochs is None:
        n_epochs = CONFIG.training.epochs
    if learning_rate is None:
        learning_rate = CONFIG.training.learning_rate
    if warmup_steps is None:
        warmup_steps = CONFIG.training.warmup_steps
    if max_grad_norm is None:
        max_grad_norm = CONFIG.training.max_grad_norm
    
    history = []
    
    print("Starting Electromagnetic PINN training with PML framework...")
    print(f"Equation: (-∇² - εω²) Ez = -iω Jz")
    print(f"Interior domain: {CONFIG.domain.interior_x} x {CONFIG.domain.interior_y}")
    print(f"Extended domain: {CONFIG.domain.extended_x} x {CONFIG.domain.extended_y}")
    print(f"PML parameters: a0={CONFIG.physics.a0}, Lpml={CONFIG.domain.lpml}")
    print(f"Gaussian current source Jz at: {CONFIG.physics.source_pos} with σ² = {CONFIG.source.sigma_squared}")
    print(f"Learning rate: {learning_rate} with {warmup_steps} warmup steps")
    print(f"Gradient clipping: max_norm = {max_grad_norm}")
    print()
    print("Theory: In interior A=B=C=1 (standard EM wave equation)")
    print("        In PML A,B,C are complex (absorbing layers)")
    print()
    
    # Create learning rate schedule with warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=n_epochs - warmup_steps,
        end_value=learning_rate * 0.1
    )
    
    # Adam optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(schedule)
    )
    opt_state = optimizer.init(params)
    
    for t in range(n_epochs):
        # Generate collocation points covering full extended domain
        circle_radius = (CONFIG.points.circle_radius_start + 
                        t/n_epochs * CONFIG.points.circle_radius_growth)
        
        x_collocation = generate_fixed_total_points(
            CONFIG.points.base_nx, 
            CONFIG.points.base_ny, 
            CONFIG.points.n_circle_points, 
            circle_radius=circle_radius, 
            random_perturbation=True, 
            key=jax.random.PRNGKey(t)
        ) 
        val, grads = value_and_grad_fn_pml(params, x_collocation)
        
        # Update parameters with clipped gradients
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        if t % 100 == 0:
            pde_loss = evaluate_pml(params, x_collocation)
            current_lr = schedule(t)
            grad_norm = optax.global_norm(grads)
            
            print(f"Epoch {t}: Total Loss = {val:.6e}  EM PDE Loss = {pde_loss:.6e}  "
                  f"LR = {current_lr:.2e}  GradNorm = {grad_norm:.4f}")
            history.append((t, float(val), float(pde_loss), float(current_lr), float(grad_norm)))
    
    return params, history