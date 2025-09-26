#!/usr/bin/env python3
"""
Main pipeline execution script for scaling laws experiments
"""

import argparse
import yaml
import os
import importlib
import sys
from datetime import datetime

def load_pipeline_config(config_path):
    """Load pipeline configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_directories(config):
    """Create necessary directories"""
    dirs_to_create = [
        config.get('results_dir', 'results'),
        config.get('logs_dir', 'logs'), 
        config.get('checkpoints_dir', 'checkpoints')
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def execute_training_step(step_config, model_size, pipeline_config):
    """Execute a single training step for a model"""
    print(f"\n{'='*60}")
    print(f"Executing step: {step_config['name']} for model: {model_size}")
    print(f"{'='*60}")
    
    # Import the training module
    module_name = step_config['module']
    function_name = step_config['function']
    
    try:
        module = importlib.import_module(module_name)
        training_function = getattr(module, function_name)
    except (ImportError, AttributeError) as e:
        print(f"Error importing {module_name}.{function_name}: {e}")
        return False
    
    # Execute the training step
    try:
        success = training_function(
            model_size=model_size,
            model_config_path=pipeline_config['model_config'],
            data_config_path=pipeline_config['data_config'],
            training_config_path=pipeline_config['training_config'],
            results_dir=pipeline_config['results_dir'],
            logs_dir=pipeline_config['logs_dir'],
            checkpoints_dir=pipeline_config['checkpoints_dir']
        )
        return success
    except Exception as e:
        print(f"Error executing {step_config['name']} for {model_size}: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_all_model_parameters(config):
    """Print parameter count for all models before training"""
    from src.models.model_factory import create_model
    
    print(f"\n{'='*60}")
    print("MODEL PARAMETER COUNTS")
    print(f"{'='*60}")
    
    models = config['models']
    model_config_path = config['model_config']
    
    total_params = 0
    for model_size in models:
        try:
            model, model_config = create_model(model_size, model_config_path)
            params = model.get_num_params()
            total_params += params
            
            print(f"{model_size.upper():>8}: {params:>15,} parameters")
            
            # Print architecture details
            arch = model_config['architecture']
            print(f"{'':>8}  └─ layers: {arch['n_layers']}, d_model: {arch['d_model']}, heads: {arch['n_heads']}")
            
        except Exception as e:
            print(f"{model_size.upper():>8}: Error creating model - {e}")
    
    print(f"{'':>8}")
    print(f"{'TOTAL':>8}: {total_params:>15,} parameters across all models")
    print(f"{'='*60}")

def run_pipeline(pipeline_config_path):
    """Run the full training pipeline"""
    # Load pipeline configuration
    config = load_pipeline_config(pipeline_config_path)
    
    print(f"Starting experiment: {config['experiment_name']}")
    print(f"Description: {config['description']}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup directories
    setup_directories(config)
    
    # Get models to train
    models = config['models']
    steps = config['steps']
    
    print(f"\nModels to train: {models}")
    print(f"Training steps: {[step['name'] for step in steps]}")
    
    # Print all model parameter counts
    print_all_model_parameters(config)
    
    # Execute pipeline for each model
    results = {}
    
    for model_size in models:
        print(f"\n{'#'*80}")
        print(f"TRAINING MODEL: {model_size.upper()}")
        print(f"{'#'*80}")
        
        model_results = {}
        
        # Execute each training step
        for step in steps:
            step_name = step['name']
            print(f"\nStarting {step_name} for {model_size}...")
            
            success = execute_training_step(step, model_size, config)
            model_results[step_name] = success
            
            if success:
                print(f" {step_name} completed successfully for {model_size}")
            else:
                print(f"L {step_name} failed for {model_size}")
                # Continue with next model instead of stopping
                break
        
        results[model_size] = model_results
    
    # Print final summary
    print(f"\n{'='*80}")
    print("PIPELINE EXECUTION SUMMARY")
    print(f"{'='*80}")
    
    for model_size, model_results in results.items():
        print(f"\nModel: {model_size}")
        for step_name, success in model_results.items():
            status = " SUCCESS" if success else "L FAILED"
            print(f"  {step_name}: {status}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run scaling laws training pipeline")
    parser.add_argument(
        "--config", 
        default="config/pipeline.yaml",
        help="Path to pipeline configuration file"
    )
    parser.add_argument(
        "--model",
        help="Train only specific model size (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Add project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Load and potentially modify config
    config = load_pipeline_config(args.config)
    
    if args.model:
        # Override models list if specific model requested
        config['models'] = [args.model]
        print(f"Training single model: {args.model}")
    
    # Run the pipeline
    try:
        results = run_pipeline(args.config)
        print(f"\n<� Pipeline execution completed!")
        return 0
    except KeyboardInterrupt:
        print(f"\n�  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n=� Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())