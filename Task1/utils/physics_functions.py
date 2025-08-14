# utils/physics_functions.py
"""Physical functions for electromagnetic wave simulation"""

import jax
import jax.numpy as jnp
from jax import vmap
from .config import CONFIG


# utils/physics_functions.py
"""Physical functions for electromagnetic wave simulation"""

import jax
import jax.numpy as jnp
from jax import vmap
from .config import CONFIG, get_source_pos


def epsilon(x):
    """Permittivity function using trained level set model"""
    try:
        from utils.permitivity_level_set import LevelSetSIREN
        
        # Load the model (you might want to cache this)
        loaded_model = LevelSetSIREN.from_file(CONFIG.paths.level_set_model)
        
        # Shift point according to config
        shifted_point = jnp.array([
            x[0] + CONFIG.permittivity.shift_x, 
            x[1] + CONFIG.permittivity.shift_y
        ])
        
        # Check if shifted point is in bounds
        in_bounds = (
            (shifted_point[0] >= CONFIG.permittivity.bounds_min) & 
            (shifted_point[0] <= CONFIG.permittivity.bounds_max) & 
            (shifted_point[1] >= CONFIG.permittivity.bounds_min) & 
            (shifted_point[1] <= CONFIG.permittivity.bounds_max)
        )
        
        # Get level set value
        level_value = loaded_model.predict_level_set(shifted_point.reshape(1, -1))[0, 0]
        
        return jax.lax.cond(
            in_bounds & (level_value < 0.),  # Inside bounds AND negative level set
            lambda: CONFIG.permittivity.inside_value,   # Inside: high permittivity
            lambda: CONFIG.permittivity.outside_value   # Outside: free space
        )
    except ImportError:
        # Fallback to simple circular permittivity if level set model not available
        source_pos = get_source_pos()
        r_squared = (x[0] - source_pos[0])**2 + (x[1] - source_pos[1])**2
        return jax.lax.cond(
            r_squared < 1.0,  # Simple circular region
            lambda: CONFIG.permittivity.inside_value,
            lambda: CONFIG.permittivity.outside_value
        )


def current_density_Jz(x):
    """
    Gaussian current density Jz(x) for the electromagnetic source
    Modeled as Gaussian with σ² from config, centered at source_pos
    """
    source_pos = get_source_pos()
    # Distance squared from source position
    r_squared = ((x[0] - source_pos[0])**2 + 
                (x[1] - source_pos[1])**2)
    
    # Gaussian source: Jz(x) = (1/(2πσ²)) * exp(-r²/(2σ²))
    normalization = 1.0 / (2 * jnp.pi * CONFIG.source.sigma_squared)
    
    # Gaussian function
    gaussian = normalization * jnp.exp(-r_squared / (2 * CONFIG.source.sigma_squared))
    
    # Apply threshold: set values < threshold to 0
    result = jnp.where(gaussian < CONFIG.source.threshold, 0.0, gaussian)
    
    return result


# Complex number operations for JAX
def complex_mul(z1, z2):
    """Multiply two complex numbers represented as [real, imag]"""
    a, b = z1[0], z1[1]
    c, d = z2[0], z2[1]
    return jnp.array([a*c - b*d, a*d + b*c])


def complex_div(z1, z2):
    """Divide two complex numbers represented as [real, imag]"""
    a, b = z1[0], z1[1]
    c, d = z2[0], z2[1]
    denom = c**2 + d**2
    return jnp.array([(a*c + b*d)/denom, (b*c - a*d)/denom])


def complex_conj(z):
    """Complex conjugate"""
    return jnp.array([z[0], -z[1]])


def compute_pml_coordinates(x):
    """
    Compute PML coordinate transformations following the theoretical framework:
    ex = 1 - i*σx/ω, ey = 1 - i*σy/ω
    where σx, σy depend on distance into PML regions
    """
    xy_in = CONFIG.domain.xy_in
    lpml = CONFIG.domain.lpml
    wavenumber = CONFIG.physics.wavenumber
    a0 = CONFIG.physics.a0
    
    # Distance from interior boundary into PML (lx, ly in the theory)
    # Left PML: x in [-(xy_in+lpml), -xy_in]
    lx_west = jnp.where(x[0] < -xy_in, -xy_in - x[0], 0.0)
    # Right PML: x in [xy_in, (xy_in+lpml)]  
    lx_east = jnp.where(x[0] > xy_in, x[0] - xy_in, 0.0)
    # Bottom PML: y in [-(xy_in+lpml), -xy_in]
    ly_south = jnp.where(x[1] < -xy_in, -xy_in - x[1], 0.0)
    # Top PML: y in [xy_in, (xy_in+lpml)]
    ly_north = jnp.where(x[1] > xy_in, x[1] - xy_in, 0.0)
    
    # PML damping functions following the theory:
    # σx = (a0*wavenumber) * (lx/lpml)² inside PML, 0 outside PML
    sigma_x = (a0 * wavenumber) * ((lx_west / lpml)**2 + (lx_east / lpml)**2)
    sigma_y = (a0 * wavenumber) * ((ly_south / lpml)**2 + (ly_north / lpml)**2)
    
    # Complex coordinate stretching factors: e = 1 - i*σ/ω
    # Note: In interior domain, σx = σy = 0, so ex = ey = 1
    ex = jnp.array([1.0, -sigma_x / wavenumber])  # [real, imag] = [1, -σx/ω]
    ey = jnp.array([1.0, -sigma_y / wavenumber])  # [real, imag] = [1, -σy/ω]
    
    # PML transformation coefficients
    # A = ey/ex, B = ex/ey, C = ex*ey
    # In interior: A = B = C = 1 (since ex = ey = 1)
    A = complex_div(ey, ex)  # ey/ex
    B = complex_div(ex, ey)  # ex/ey  
    C = complex_mul(ex, ey)  # ex*ey
    
    return A, B, C