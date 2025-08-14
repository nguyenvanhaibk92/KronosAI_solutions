# utils/config.py
"""Configuration management using ml_collections and YAML"""

from pathlib import Path
import ml_collections
import yaml
import jax.numpy as jnp


def get_config():
    """Loads configuration from YAML file and returns a ConfigDict."""
    # Load and parse YAML file
    with open('utils/config.yaml', 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Initialize ConfigDict
    config = ml_collections.ConfigDict()
    
    # Transfer all values from YAML to ConfigDict
    for key, value in yaml_config.items():
        if isinstance(value, dict):
            config[key] = ml_collections.ConfigDict(value)
        else:
            config[key] = value
    
    # Calculate computed parameters
    # Physical computational domain (interior)
    config.domain.interior_x = (-config.domain.xy_in, config.domain.xy_in)
    config.domain.interior_y = (-config.domain.xy_in, config.domain.xy_in)
    
    # Extended domain including PML layers
    config.domain.extended_x = (
        -config.domain.xy_in - config.domain.lpml, 
        config.domain.xy_in + config.domain.lpml
    )
    config.domain.extended_y = (
        -config.domain.xy_in - config.domain.lpml, 
        config.domain.xy_in + config.domain.lpml
    )
    
    # Set wavenumber equal to omega if not specified differently
    if config.physics.wavenumber is None:
        config.physics.wavenumber = config.physics.omega
    
    return config


def get_source_pos():
    """Get source position as JAX array"""
    return jnp.array(CONFIG.physics.source_pos)


# Global config instance - load once when module is imported
CONFIG = get_config()