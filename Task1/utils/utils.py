# utils/utils.py
"""Utility functions for the electromagnetic PINN simulation"""

import jax.numpy as jnp
import numpy as np
import pickle
from .config import CONFIG


def save_model(params, filename):
    """Save model parameters to file"""
    with open(filename, 'wb') as f:
        pickle.dump(params, f)
    print(f"Model saved to {filename}")


def load_model(filename):
    """Load model parameters from file"""
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    print(f"Model loaded from {filename}")
    return params


def save_training_history(history, filename):
    """Save training history to file"""
    np.save(filename, np.array(history))
    print(f"Training history saved to {filename}")


def load_training_history(filename):
    """Load training history from file"""
    history = np.load(filename)
    print(f"Training history loaded from {filename}")
    return history.tolist()


def compute_field_magnitude(Ez_complex):
    """Compute magnitude of complex electromagnetic field"""
    real_part = Ez_complex[..., 0]
    imag_part = Ez_complex[..., 1]
    return jnp.sqrt(real_part**2 + imag_part**2)


def compute_field_phase(Ez_complex):
    """Compute phase of complex electromagnetic field"""
    real_part = Ez_complex[..., 0]
    imag_part = Ez_complex[..., 1]
    return jnp.arctan2(imag_part, real_part)


def print_model_info(params):
    """Print information about model parameters"""
    params1, params2 = params
    
    print("Model Information:")
    print("-" * 30)
    print(f"Number of parameter sets: 2 (real and imaginary)")
    print(f"Real network layers: {len(params1)}")
    print(f"Imaginary network layers: {len(params2)}")
    
    total_params = 0
    for i, (layer1, layer2) in enumerate(zip(params1, params2)):
        weights1, biases1 = layer1
        weights2, biases2 = layer2
        layer_params = weights1.size + biases1.size + weights2.size + biases2.size
        total_params += layer_params
        print(f"Layer {i+1}: {weights1.shape} weights + {biases1.shape} biases (×2 networks)")
    
    print(f"Total parameters: {total_params:,}")


def create_circular_mask(X, Y, center, radius):
    """Create a circular mask for visualization"""
    distances = jnp.sqrt((X - center[0])**2 + (Y - center[1])**2)
    return distances <= radius


def apply_pml_visualization_mask(X, Y):
    """Create a mask to highlight PML regions"""
    interior_x = CONFIG.domain.interior_x
    interior_y = CONFIG.domain.interior_y
    
    interior_mask = ((X >= interior_x[0]) & (X <= interior_x[1]) & 
                    (Y >= interior_y[0]) & (Y <= interior_y[1]))
    return ~interior_mask  # PML regions are outside interior


def get_domain_info():
    """Get information about computational domains"""
    info = {
        'extended_domain': {
            'x': CONFIG.domain.extended_x,
            'y': CONFIG.domain.extended_y,
            'area': ((CONFIG.domain.extended_x[1] - CONFIG.domain.extended_x[0]) * 
                    (CONFIG.domain.extended_y[1] - CONFIG.domain.extended_y[0]))
        },
        'interior_domain': {
            'x': CONFIG.domain.interior_x,
            'y': CONFIG.domain.interior_y,
            'area': ((CONFIG.domain.interior_x[1] - CONFIG.domain.interior_x[0]) * 
                    (CONFIG.domain.interior_y[1] - CONFIG.domain.interior_y[0]))
        },
        'pml_thickness': CONFIG.domain.lpml,
        'source_position': CONFIG.physics.source_pos
    }
    
    return info


def print_config_summary():
    """Print a summary of the current configuration"""
    print("Configuration Summary:")
    print("=" * 50)
    print(f"Domain: {CONFIG.domain.extended_x} x {CONFIG.domain.extended_y}")
    print(f"Interior: {CONFIG.domain.interior_x} x {CONFIG.domain.interior_y}")
    print(f"PML thickness: {CONFIG.domain.lpml}")
    print(f"Source position: {CONFIG.physics.source_pos}")
    print(f"Frequency (ω): {CONFIG.physics.omega}")
    print(f"Network architecture: {CONFIG.network.neurons}")
    print(f"Training epochs: {CONFIG.training.epochs}")
    print(f"Learning rate: {CONFIG.training.learning_rate}")
    print("=" * 50)