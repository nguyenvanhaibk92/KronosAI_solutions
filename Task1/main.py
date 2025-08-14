# %%
# main_two_phase.py
"""Main script demonstrating two-phase electromagnetic PINN training"""

import os
# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


import jax
from jax import random

# Import utils modules
from utils import (
    CONFIG, init_mlp_params, print_config_summary, print_model_info,
    save_model, load_model, get_domain_info,
    initialize_networks, 
    visualize_permittivity, visualize_points_distribution, 
    run_two_phase_training, 
    net_with_bc, net_with_bc_phase2,
    visualize_solution, visualize_solution_phase2
)


# %%

"""Main two-phase training and visualization pipeline"""
print("Two-Phase Electromagnetic PINN Simulation")
print("=" * 60)

# Print configuration summary
print_config_summary()

# Print domain information
domain_info = get_domain_info()
print(f"\nDomain Information:")
print(f"Extended domain area: {domain_info['extended_domain']['area']:.2f}")
print(f"Interior domain area: {domain_info['interior_domain']['area']:.2f}")
print(f"PML coverage: {(1 - domain_info['interior_domain']['area']/domain_info['extended_domain']['area'])*100:.1f}%")

# Initialize random key
rng = random.PRNGKey(42)

# Initialize networks
print("\nInitializing SIREN networks...")
params_initial = initialize_networks(rng)
print_model_info(params_initial)

# Visualize setup
print("\nVisualizing problem setup...")
visualize_permittivity()
pts = visualize_points_distribution()
print(f"Generated {pts.shape[0]} training points")

# Run two-phase training
print("\nStarting Two-Phase Training...")
params_phase1, params_phase2, history_phase1, history_phase2 = run_two_phase_training(params_initial)


# %%
# Save both phases
print("\nSaving trained models...")
save_model(params_phase1, "models/trained_em_pinn_phase1.pkl")
save_model(params_phase2, "models/trained_em_pinn_phase2.pkl")

