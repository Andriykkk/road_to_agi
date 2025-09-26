# Project Structure Documentation

## Overview
This project is designed for training models of different sizes with flexible configuration management, data sampling, and pluggable training methods.

## Directory Structure

### `config/`
- **models.yaml** - Model architecture definitions for different sizes (tiny, small, medium, large)
- **training.yaml** - Hyperparameters and training configurations for each model size
- **data.yaml** - Data sampling configurations (percentages, intervals, preprocessing)
- **pipeline.yaml** - Training pipeline steps and their execution order

### `src/data/`
- **data_loader.py** - Handles loading and preprocessing of training data
- **sampler.py** - Implements flexible sampling strategies (percentage, interval-based)

### `src/models/`
- **architectures.py** - Base model architectures and scaling rules
- **model_factory.py** - Factory for creating models based on configuration

### `src/training/`
- **pretraining_diffusion.py** - Pretraining module for diffusion models
- **pretraining_regression.py** - Pretraining module for regression tasks
- **reinforcement_learning.py** - RL training implementation
- **fine_tuning.py** - Fine-tuning training module

### `src/pipeline/`
- **orchestrator.py** - Main pipeline orchestrator that runs training steps
- **experiment_tracker.py** - Tracks experiments, logs, and results

### Root Files
- **main.py** - Main entry point for running experiments
- **requirements.txt** - Project dependencies
- **logs/** - Training logs organized by experiment
- **experiments/** - Experiment configurations and results
- **results/** - Model outputs, checkpoints, and analysis

## Usage Flow
1. Configure model sizes in `config/models.yaml`
2. Set hyperparameters in `config/training.yaml`
3. Define data sampling in `config/data.yaml`
4. Configure training pipeline in `config/pipeline.yaml`
5. Run `python main.py` to execute the full pipeline
6. Results and logs are saved to respective directories for analysis and scaling extrapolation