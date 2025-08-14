# utils/pinn_model.py
"""PINN model implementation for electromagnetic wave equation with PML"""

import jax
import jax.numpy as jnp
from jax import vmap
from .siren_network import SIREN_neural_one_sample
from .physics_functions import (
    compute_pml_coordinates, complex_mul, epsilon, current_density_Jz
)
from .config import CONFIG, get_source_pos


def net_with_bc(x, params):
    """Network output with boundary conditions"""
    source_pos = get_source_pos()
    x = (x - source_pos)  # Center the input around the source position
    params1, params2 = params
    re = SIREN_neural_one_sample(x, params1).squeeze()
    im = SIREN_neural_one_sample(x, params2).squeeze()
    return jnp.array([re, im])  # Return as complex array [re, im]


def net_grad(x, params):
    """Compute gradients of both real and imaginary parts efficiently"""
    grad_fn = jax.jacfwd(net_with_bc, argnums=0)  # Jacobian w.r.t. x
    return grad_fn(x, params)  # Shape: [2, 2] -> [[dre/dx, dre/dy], [dim/dx, dim/dy]]


def compute_pml_derivatives(x, params, A, B):
    """Compute both A*∂Ez/∂x and B*∂Ez/∂y efficiently"""
    grad_Ez = net_grad(x, params)  # [2, 2] array
    
    # A * ∂Ez/∂x for both real and imaginary parts
    A_dEz_dx = complex_mul(A, grad_Ez[:, 0])  # [:, 0] is ∂/∂x
    
    # B * ∂Ez/∂y for both real and imaginary parts  
    B_dEz_dy = complex_mul(B, grad_Ez[:, 1])  # [:, 1] is ∂/∂y
    
    return A_dEz_dx, B_dEz_dy


def compute_divergence_terms(x, params, A, B):
    """Compute ∂/∂x(A*∂Ez/∂x) + ∂/∂y(B*∂Ez/∂y) efficiently"""
    
    # Define functions for second derivatives
    def A_term(pt, A):
        grad_Ez = net_grad(pt, params)
        return complex_mul(A, grad_Ez[:, 0])  # A * ∂Ez/∂x
    
    def B_term(pt, B):
        grad_Ez = net_grad(pt, params)
        return complex_mul(B, grad_Ez[:, 1])  # B * ∂Ez/∂y
    
    # Compute divergence using Jacobian: ∂/∂x(A*∂Ez/∂x) + ∂/∂y(B*∂Ez/∂y)
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
def electromagnetic_pml_residual_single(params, x):
    """
    Condensed electromagnetic wave equation with PML:
    ∂/∂x(A ∂Ez/∂x) + ∂/∂y(B ∂Ez/∂y) + C εω² Ez = iω Jz
    """
    # Get PML coefficients and field
    A, B, C = compute_pml_coordinates(x)
    Ez = net_with_bc(x, params)
    
    # Compute divergence terms efficiently
    div_term = compute_divergence_terms(x, params, A, B)
    
    # εω² term with C coefficient
    wavenumber = CONFIG.physics.wavenumber
    C_eps_omega2_Ez = complex_mul(C, wavenumber**2 * epsilon(x) * Ez)
    
    # Source term: iω Jz = [0, ω Jz] (purely imaginary)
    i_omega_Jz = jnp.array([1.0, 1.0]) * current_density_Jz(x)
    
    # Complete residual
    residual = div_term + C_eps_omega2_Ez - i_omega_Jz
    
    return residual[0], residual[1]


@jax.jit
def pde_residual_single(params, x):
    """
    Electromagnetic wave equation residual with PML
    """
    return electromagnetic_pml_residual_single(params, x)


@jax.jit
def pinn_loss_pml(params, x_collocation):
    """
    Unified PINN loss using the Helmholtz-PML framework.
    The same equation applies everywhere with A=B=C=1 in interior 
    and complex A,B,C in PML regions.
    """
    # PDE residuals for all points using unified framework
    residuals_re, residuals_im = vmap(lambda x: pde_residual_single(params, x))(x_collocation)
    squared_residuals = jnp.abs(residuals_re) + jnp.abs(residuals_im)
    
    # Single unified loss (no need to separate interior vs PML)
    pde_loss = jnp.mean(squared_residuals)
    
    return pde_loss


@jax.jit
def evaluate_pml(params, x_collocation):
    """Evaluation function"""
    pde_loss = pinn_loss_pml(params, x_collocation)
    return pde_loss


@jax.jit
def value_and_grad_fn_pml(params, x_collocation):
    """Value and gradient function"""
    return jax.value_and_grad(lambda p: pinn_loss_pml(p, x_collocation))(params)