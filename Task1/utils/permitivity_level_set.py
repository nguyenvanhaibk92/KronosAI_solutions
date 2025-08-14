import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path  # For SVG path parsing
import time
import optax
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random, value_and_grad, hessian
import pickle
import json
from pathlib import Path as FilePath  # For file system paths

# Configuration
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = "4.0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class LevelSetSIREN:
    def __init__(self, omega_0=30, neurons=[2, 8, 1], mesh_constraints=[32, 32]):
        self.omega_0 = omega_0
        self.neurons = neurons
        self.mesh_constraints = np.asarray(mesh_constraints)
        self.rng = random.PRNGKey(0)
        
    def init_mlp_params(self, layer_widths):
        """Initialize SIREN network parameters"""
        params = []
        for i, (n_in, n_out) in enumerate(zip(layer_widths[:-1], layer_widths[1:])):
            if i == 0:
                # First layer initialization
                params.append([
                    jax.random.uniform(self.rng, shape=(n_in, n_out),
                                     minval=-(1/n_in), maxval=(1/n_in)),
                    jax.random.uniform(self.rng, shape=(n_out,),
                                     minval=-(1/n_in)/self.omega_0, 
                                     maxval=(1/n_in)/self.omega_0)
                ])
            else:
                # Hidden layers initialization
                params.append([
                    jax.random.uniform(random.PRNGKey(i), shape=(n_in, n_out),
                                     minval=-jnp.sqrt(6/n_in)/self.omega_0,
                                     maxval=jnp.sqrt(6/n_in)/self.omega_0),
                    jax.random.uniform(random.PRNGKey(i), shape=(n_out,),
                                     minval=-jnp.sqrt(6/n_in)/self.omega_0,
                                     maxval=jnp.sqrt(6/n_in)/self.omega_0)
                ])
        return params
    
    def siren_forward_single(self, x_input, params):
        """Forward pass for single input"""
        x = jnp.sin(self.omega_0 * (x_input @ params[0][0] + params[0][1]))
        for i in range(1, len(params)-1):
            x = jnp.sin(self.omega_0 * (x @ params[i][0] + params[i][1]))
        return (x @ params[-1][0]) + params[-1][1]
    
    def setup_network_functions(self):
        """Setup JIT compiled network functions"""
        self.siren_forward = jax.jit(vmap(self.siren_forward_single, in_axes=(0, None)))
        
        def siren_predict(x_input, params):
            return self.siren_forward_single(x_input, params).mean()
        
        def gradient_magnitude_single(x, params):
            return jnp.square(jax.grad(siren_predict)(x, params)).sum()
        
        def hessian_single(x, params):
            return hessian(siren_predict)(x, params)
        
        self.gradient_magnitude = jax.jit(vmap(gradient_magnitude_single, in_axes=(0, None)))
        self.hessian_batch = jax.jit(vmap(hessian_single, in_axes=(0, None)))
    
    def prepare_data_constraints(self, negative_points, positive_points):
        """Prepare constraint points for gradient computation"""
        # Combine negative and positive points for data points
        data_points = np.vstack([negative_points, positive_points])
        
        # Create constraint grid for gradient computation
        # Use the bounding box of all points
        all_points = data_points
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        
        # Expand bounds slightly
        x_range = x_max - x_min
        y_range = y_max - y_min
        margin = 0.2 * max(x_range, y_range)
        
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        
        # Create constraint grid
        x = np.linspace(x_min, x_max, self.mesh_constraints[0])
        y = np.linspace(y_min, y_max, self.mesh_constraints[1])
        x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
        constraint_points = np.concatenate((x_grid.flatten(), y_grid.flatten()), axis=0).reshape(2, -1).T
        
        # Gradient constraint points (middle points)
        x_mid = (x[:-1] + x[1:]) / 2
        y_mid = (y[:-1] + y[1:]) / 2
        x_mid_grid, y_mid_grid = np.meshgrid(x_mid, y_mid, indexing='ij')
        grad_points = np.concatenate((x_mid_grid.flatten(), y_mid_grid.flatten()), axis=0).reshape(2, -1).T
        
        return {
            'data_points': data_points,
            'negative_points': negative_points,
            'positive_points': positive_points,
            'constraint_points': constraint_points,
            'gradient_points': grad_points,
            'hessian_points': constraint_points
        }
    
    def setup_loss_functions(self):
        """Setup loss functions for training - only smoothness, magnitude, and gradient"""
        @jit
        def loss_gradient(params, constraint_points):
            """Signed distance constraint: |∇φ| = 1"""
            return jnp.mean((self.gradient_magnitude(constraint_points, params) - 1)**2)
        
        @jit
        def loss_magnitude(params, negative_points, positive_points):
            """Sign constraint for level set orientation"""
            sign_negative = self.siren_forward(negative_points, params)
            sign_positive = self.siren_forward(positive_points, params)
            return (jnp.mean(jnp.maximum(sign_negative, 0)**2) + 
                   jnp.mean(jnp.maximum(-sign_positive, 0)**2))
        
        @jit
        def loss_smoothness(params, constraint_points):
            """Smoothness regularization"""
            return jnp.mean(self.hessian_batch(constraint_points, params)**2)
        
        @jit
        def total_loss(params, constraints, alpha_gradient, alpha_smoothness):
            L_gradient = loss_gradient(params, constraints['gradient_points'])
            L_magnitude = loss_magnitude(params, constraints['negative_points'], 
                                       constraints['positive_points'])
            L_smoothness = loss_smoothness(params, constraints['constraint_points'])
            
            return (L_magnitude + alpha_gradient * L_gradient + 
                   alpha_smoothness * L_smoothness)
        
        self.loss_functions = {
            'gradient': loss_gradient,
            'magnitude': loss_magnitude,
            'smoothness': loss_smoothness,
            'total': total_loss
        }
    
    def train(self, negative_points, positive_points, alpha_gradient=1e-2, alpha_smoothness=1e-4, num_epochs=2000):
        """Train the level set network with negative and positive points"""
        # Initialize network
        params = self.init_mlp_params(self.neurons)
        self.setup_network_functions()
        self.setup_loss_functions()
        
        # Prepare constraints
        constraints = self.prepare_data_constraints(negative_points, positive_points)
        
        # Setup optimizer
        solver = optax.lbfgs()
        opt_state = solver.init(params)
        
        def objective(x):
            return self.loss_functions['total'](x, constraints, alpha_gradient, alpha_smoothness)
        
        value_and_grad_fn = optax.value_and_grad_from_state(objective)
        
        @jax.jit
        def update_step(params, opt_state):
            value, grad = value_and_grad_fn(params, state=opt_state)
            updates, opt_state = solver.update(
                grad, opt_state, params, value=value, grad=grad, value_fn=objective
            )
            params = optax.apply_updates(params, updates)
            return params, opt_state
        
        # Training loop
        history = {'magnitude': [], 'gradient': [], 'smoothness': []}
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            params, opt_state = update_step(params, opt_state)
            epoch_time = time.time() - start_time
            
            if (epoch - 1) % 250 == 0:
                # Compute losses for logging
                L_magnitude = self.loss_functions['magnitude'](params, negative_points, positive_points)
                
                L_gradient = self.loss_functions['gradient'](params, constraints['gradient_points'])
                
                L_smoothness = self.loss_functions['smoothness'](params, constraints['constraint_points'])
                
                print(f"Epoch: {epoch:5d}, time: {epoch_time:.4f}, "
                      f"L_magnitude: {L_magnitude:.4e}, "
                      f"L_gradient: {L_gradient:.4e}, "
                      f"L_smoothness: {L_smoothness:.4e}")
                
                history['magnitude'].append(L_magnitude)
                history['gradient'].append(L_gradient)
                history['smoothness'].append(L_smoothness)
        
        self.trained_params = params
        self.training_history = history
        self.data_points = constraints['data_points']  # Store for plotting
        return params
    
    def evaluate_level_set(self, x_range=(-1, 1), y_range=(-1, 1), resolution=64):
        """Evaluate the learned level set on a grid"""
        if not hasattr(self, 'trained_params'):
            raise ValueError("Network must be trained first!")
        
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
        grid_points = np.concatenate((x_grid.flatten(), y_grid.flatten()), axis=0).reshape(2, -1).T
        
        phi_values = self.siren_forward(grid_points, self.trained_params)
        return x_grid, y_grid, phi_values.reshape(resolution, resolution)
    
    def predict_level_set(self, points):
        """Predict level set values for given points"""
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        return self.siren_forward(points, self.trained_params)

    def plot_results(self, negative_points=None, positive_points=None, x_range=None, y_range=None, levels=None):
        """Plot the learned level set"""
        # Use stored data points if not provided
        if negative_points is None or positive_points is None:
            if hasattr(self, 'data_points'):
                # Assume first half are negative, second half are positive (this is a simplification)
                mid_point = len(self.data_points) // 2
                if negative_points is None:
                    negative_points = self.data_points[:mid_point]
                if positive_points is None:
                    positive_points = self.data_points[mid_point:]
            else:
                raise ValueError("Must provide negative_points and positive_points or have trained data stored")
        
        # Auto-determine plot range if not provided
        if x_range is None or y_range is None:
            all_points = np.vstack([negative_points, positive_points])
            x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
            y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
            
            # Add margin
            x_range_val = x_max - x_min
            y_range_val = y_max - y_min
            margin = 0.3 * max(x_range_val, y_range_val)
            
            if x_range is None:
                x_range = (x_min - margin, x_max + margin)
            if y_range is None:
                y_range = (y_min - margin, y_max + margin)
        
        x_grid, y_grid, phi = self.evaluate_level_set(x_range, y_range)
        
        if levels is None:
            phi_range = np.abs(phi).max()
            levels = np.linspace(-phi_range, phi_range, 9)
        
        plt.figure(figsize=(10, 8))
        
        # Plot level set contours
        contour = plt.contour(x_grid, y_grid, phi, colors='k', levels=levels)
        plt.clabel(contour, inline=True, fontsize=10)
        
        # Highlight zero level set
        zero_contour = plt.contour(x_grid, y_grid, phi, colors='red', levels=[0], linewidths=3)
        
        # Plot negative points (inside)
        plt.scatter(negative_points[:, 0], negative_points[:, 1], c='blue', s=40, 
                   label='Negative Points (Inside)', marker='o')
        
        # Plot positive points (outside)
        plt.scatter(positive_points[:, 0], positive_points[:, 1], c='red', s=40, 
                   label='Positive Points (Outside)', marker='s')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.legend()
        plt.title('Learned Level Set (Red line = Zero Level Set)')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model parameters and configuration"""
        if not hasattr(self, 'trained_params'):
            raise ValueError("No trained parameters to save. Train the model first!")
        
        filepath = FilePath(filepath)  # Fixed: Use FilePath instead of Path
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert JAX arrays to numpy for serialization
        def jax_to_numpy(params):
            if isinstance(params, list):
                return [jax_to_numpy(p) for p in params]
            elif isinstance(params, jnp.ndarray):
                return np.array(params)
            else:
                return params
        
        save_data = {
            'trained_params': jax_to_numpy(self.trained_params),
            'omega_0': self.omega_0,
            'neurons': self.neurons,
            'mesh_constraints': self.mesh_constraints.tolist(),
            'training_history': self.training_history if hasattr(self, 'training_history') else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from file"""
        filepath = FilePath(filepath)  # Fixed: Use FilePath instead of Path
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Convert numpy arrays back to JAX arrays
        def numpy_to_jax(params):
            if isinstance(params, list):
                return [numpy_to_jax(p) for p in params]
            elif isinstance(params, np.ndarray):
                return jnp.array(params)
            else:
                return params
        
        self.trained_params = numpy_to_jax(save_data['trained_params'])
        self.omega_0 = save_data['omega_0']
        self.neurons = save_data['neurons']
        self.mesh_constraints = np.array(save_data['mesh_constraints'])
        
        if save_data.get('training_history'):
            self.training_history = save_data['training_history']
        
        # Reinitialize network functions with loaded parameters
        self.setup_network_functions()
        
        print(f"Model loaded from {filepath}")
        return self
    
    @classmethod
    def from_file(cls, filepath):
        """Create a new instance and load model from file"""
        instance = cls()
        instance.load_model(filepath)
        return instance
    
    

def get_dolphin_points(mesh_density=250):
    """
    Returns inside and outside points for the dolphin shape in domain [-1,1]^2
    
    Parameters:
    mesh_density: int, grid resolution (default 250)
    
    Returns:
    tuple: (inside_points, outside_points) as numpy arrays
    """
    
    # Dolphin SVG path data
    dolphin = """
    M -0.59739425,160.18173 C -0.62740401,160.18885 -0.57867129,160.11183
    -0.57867129,160.11183 C -0.57867129,160.11183 -0.5438361,159.89315
    -0.39514638,159.81496 C -0.24645668,159.73678 -0.18316813,159.71981
    -0.18316813,159.71981 C -0.18316813,159.71981 -0.10322971,159.58124
    -0.057804323,159.58725 C -0.029723983,159.58913 -0.061841603,159.60356
    -0.071265813,159.62815 C -0.080250183,159.65325 -0.082918513,159.70554
    -0.061841203,159.71248 C -0.040763903,159.7194 -0.0066711426,159.71091
    0.077336307,159.73612 C 0.16879567,159.76377 0.28380306,159.86448
    0.31516668,159.91533 C 0.3465303,159.96618 0.5011127,160.1771
    0.5011127,160.1771 C 0.63668998,160.19238 0.67763022,160.31259
    0.66556395,160.32668 C 0.65339985,160.34212 0.66350443,160.33642
    0.64907098,160.33088 C 0.63463742,160.32533 0.61309688,160.297
    0.5789627,160.29339 C 0.54348657,160.28968 0.52329693,160.27674
    0.50728856,160.27737 C 0.49060916,160.27795 0.48965803,160.31565
    0.46114204,160.33673 C 0.43329696,160.35786 0.4570711,160.39871
    0.43309565,160.40685 C 0.4105108,160.41442 0.39416631,160.33027
    0.3954995,160.2935 C 0.39683269,160.25672 0.43807996,160.21522
    0.44567915,160.19734 C 0.45327833,160.17946 0.27946869,159.9424
    -0.061852613,159.99845 C -0.083965233,160.0427 -0.26176109,160.06683
    -0.26176109,160.06683 C -0.30127962,160.07028 -0.21167141,160.09731
    -0.24649368,160.1011 C -0.32642366,160.11569 -0.34521187,160.06895
    -0.40622293,160.0819 C -0.467234,160.09485 -0.56738444,160.17461
    -0.59739425,160.18173
    """
    
    # Parse the SVG path
    vertices = []
    codes = []
    parts = dolphin.split()
    i = 0
    code_map = {
        'M': Path.MOVETO,
        'C': Path.CURVE4,
        'L': Path.LINETO,
    }
    
    while i < len(parts):
        path_code = code_map[parts[i]]
        npoints = Path.NUM_VERTICES_FOR_CODE[path_code]
        codes.extend([path_code] * npoints)
        vertices.extend([[*map(float, y.split(','))]
                         for y in parts[i + 1:][:npoints]])
        i += npoints + 1
    
    vertices = np.array(vertices)
    vertices[:, 1] -= 160  # Adjust y-coordinates
    dolphin_path = Path(vertices, codes)
    
    def extract_boundary_with_interpolation(path, resolution=2000):
        """Extract boundary by densely sampling the path and interpolating curves"""
        boundary_coords = []
        current_pos = np.array([0.0, 0.0])
        
        i = 0
        while i < len(path.codes):
            code = path.codes[i]
            
            if code == Path.MOVETO:
                current_pos = path.vertices[i]
                boundary_coords.append(current_pos.copy())
                i += 1
                
            elif code == Path.LINETO:
                end_pos = path.vertices[i]
                t_vals = np.linspace(0, 1, max(10, int(resolution/50)))
                for t in t_vals[1:]:
                    point = current_pos + t * (end_pos - current_pos)
                    boundary_coords.append(point)
                current_pos = end_pos
                i += 1
                
            elif code == Path.CURVE4:
                if i + 2 < len(path.vertices):
                    p0 = current_pos
                    p1 = path.vertices[i]
                    p2 = path.vertices[i + 1] 
                    p3 = path.vertices[i + 2]
                    
                    t_vals = np.linspace(0, 1, max(20, int(resolution/25)))
                    for t in t_vals[1:]:
                        point = ((1-t)**3 * p0 + 
                                3*(1-t)**2*t * p1 + 
                                3*(1-t)*t**2 * p2 + 
                                t**3 * p3)
                        boundary_coords.append(point)
                    
                    current_pos = p3
                    i += 3
                else:
                    i += 1
            else:
                i += 1
        
        return np.array(boundary_coords)
    
    def point_in_polygon_winding(point, polygon):
        """Determine if a point is inside a polygon using winding number algorithm"""
        x, y = point
        n = len(polygon)
        winding_number = 0
        
        for i in range(n):
            xi, yi = polygon[i]
            xi1, yi1 = polygon[(i + 1) % n]
            
            if yi <= y:
                if yi1 > y:
                    if ((xi1 - xi) * (y - yi) - (x - xi) * (yi1 - yi)) > 0:
                        winding_number += 1
            else:
                if yi1 <= y:
                    if ((xi1 - xi) * (y - yi) - (x - xi) * (yi1 - yi)) < 0:
                        winding_number -= 1
        
        return winding_number != 0
    
    # Extract smooth boundary
    smooth_boundary = extract_boundary_with_interpolation(dolphin_path, resolution=3000)
    
    # Generate mesh grid in [-1,1]^2
    x_range = np.linspace(-1., 1., mesh_density)
    y_range = np.linspace(-1., 1., mesh_density)
    X, Y = np.meshgrid(x_range, y_range)
    mesh_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Classify points using smooth boundary method
    inside_mask = np.array([
        point_in_polygon_winding(point, smooth_boundary) 
        for point in mesh_points
    ])
    
    inside_points = mesh_points[inside_mask]
    outside_points = mesh_points[~inside_mask]
    
    return inside_points, outside_points