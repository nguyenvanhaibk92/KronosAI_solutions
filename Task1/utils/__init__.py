# utils/__init__.py
"""Utils package for electromagnetic PINN simulation"""

from .config import get_config, CONFIG, get_source_pos
from .siren_network import init_mlp_params, SIREN_neural_one_sample, SIREN_neural
from .point_generation import generate_fixed_total_points
from .physics_functions import (
    epsilon, current_density_Jz, complex_mul, complex_div, complex_conj,
    compute_pml_coordinates
)
from .pinn_model import (
    net_with_bc, net_grad, compute_pml_derivatives, compute_divergence_terms,
    electromagnetic_pml_residual_single, pde_residual_single, pinn_loss_pml,
    evaluate_pml, value_and_grad_fn_pml
)
from .training import train_pml
from .phase2_training import (
    net_with_bc_phase2, train_pml_phase2, combined_net_with_bc,
    run_two_phase_training, evaluate_residual_field
)# utils/__init__.py
"""Utils package for electromagnetic PINN simulation"""

from .config import get_config, CONFIG, get_source_pos
from .siren_network import (init_mlp_params, SIREN_neural_one_sample, 
                            initialize_networks, SIREN_neural)
from .point_generation import generate_fixed_total_points
from .physics_functions import (
    epsilon, current_density_Jz, complex_mul, complex_div, complex_conj,
    compute_pml_coordinates
)
from .pinn_model import (
    net_with_bc, net_grad, compute_pml_derivatives, compute_divergence_terms,
    electromagnetic_pml_residual_single, pde_residual_single, pinn_loss_pml,
    evaluate_pml, value_and_grad_fn_pml
)
from .training import train_pml
from .utils import (
    save_model, load_model, save_training_history, load_training_history,
    compute_field_magnitude, compute_field_phase, print_model_info,
    create_circular_mask, apply_pml_visualization_mask, get_domain_info,
    print_config_summary
)
from .plots import (
    visualize_permittivity, visualize_points_distribution, visualize_solution,
    plot_training_history, visualize_field_components_separately, save_field_plots,
    visualize_solution_phase2,
)

__all__ = [
    # Config
    'get_config', 'CONFIG', 'get_source_pos',
    
    # Network
    'init_mlp_params', 'SIREN_neural_one_sample', 'SIREN_neural',
    
    # Point generation
    'generate_fixed_total_points',
    
    # Physics
    'epsilon', 'current_density_Jz', 'complex_mul', 'complex_div', 'complex_conj',
    'compute_pml_coordinates',
    
    # PINN model (Phase 1)
    'net_with_bc', 'net_grad', 'compute_pml_derivatives', 'compute_divergence_terms',
    'electromagnetic_pml_residual_single', 'pde_residual_single', 'pinn_loss_pml',
    'evaluate_pml', 'value_and_grad_fn_pml',
    
    # Training
    'train_pml',
    
    # Phase 2 training
    'net_with_bc_phase2', 'train_pml_phase2', 'combined_net_with_bc',
    'run_two_phase_training', 'evaluate_residual_field',
    
    # Utilities
    'save_model', 'load_model', 'save_training_history', 'load_training_history',
    'compute_field_magnitude', 'compute_field_phase', 'print_model_info',
    'create_circular_mask', 'apply_pml_visualization_mask', 'get_domain_info',
    'print_config_summary',
    
    # Visualization
    'visualize_permittivity', 'visualize_points_distribution', 'visualize_solution',
    'plot_training_history', 'visualize_field_components_separately', 'save_field_plots',
]