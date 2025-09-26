import yaml
from .architectures import create_gpt_model

def load_model_config(config_path):
    """Load model configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_model(model_size, config_path='config/models.yaml'):
    """Create model instance from configuration
    
    Args:
        model_size: Size of model ('tiny', 'small', 'medium', 'large')
        config_path: Path to model configuration file
        
    Returns:
        Instantiated model
    """
    config = load_model_config(config_path)
    
    if model_size not in config['models']:
        raise ValueError(f"Model size '{model_size}' not found in config. Available: {list(config['models'].keys())}")
    
    model_config = config['models'][model_size]
    arch_config = model_config['architecture']
    
    # Create the model
    model = create_gpt_model(arch_config)
    
    print(f"Created {model_size} model with {model.get_num_params():,} parameters")
    print(f"Architecture: {arch_config}")
    
    return model, model_config

def get_model_info(model_size, config_path='config/models.yaml'):
    """Get model information without instantiating it
    
    Args:
        model_size: Size of model
        config_path: Path to model configuration file
        
    Returns:
        Dictionary with model information
    """
    config = load_model_config(config_path)
    
    if model_size not in config['models']:
        raise ValueError(f"Model size '{model_size}' not found in config")
    
    return config['models'][model_size]

def list_available_models(config_path='config/models.yaml'):
    """List all available model sizes"""
    config = load_model_config(config_path)
    return list(config['models'].keys())