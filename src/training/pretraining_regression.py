"""
Pretraining module for language modeling (next token prediction)
"""

import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
from tqdm import tqdm
from datetime import datetime
import wandb

from src.models.model_factory import create_model, load_model_config
from src.data.data_loader import create_full_dataset
from src.data.sampler import create_sampled_dataset

def load_configs(model_config_path, data_config_path, training_config_path):
    """Load all configuration files"""
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    with open(training_config_path, 'r') as f:
        training_config = yaml.safe_load(f)
    return model_config, data_config, training_config

def setup_logging(model_size, training_config, logs_dir):
    """Setup logging with wandb"""
    if training_config.get('use_wandb', True):
        run_name = f"{model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=training_config.get('wandb_project', 'scaling-laws'),
            entity=training_config.get('wandb_entity'),
            name=run_name,
            tags=[model_size, 'pretraining'],
            dir=logs_dir
        )
        return True
    return False

def create_optimizer_and_scheduler(model, model_config, total_steps):
    """Create optimizer and learning rate scheduler"""
    optimizer = AdamW(
        model.parameters(),
        lr=float(model_config['learning_rate']),
        weight_decay=float(model_config.get('weight_decay', 0.1))
    )
    
    scheduler_type = model_config.get('lr_scheduler', 'cosine')
    warmup_steps = int(model_config.get('warmup_steps', 1000))
    
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    else:
        scheduler = None
        
    return optimizer, scheduler, warmup_steps

def warmup_lr(optimizer, step, warmup_steps, base_lr):
    """Apply learning rate warmup"""
    if step < warmup_steps:
        lr = float(base_lr) * (step + 1) / warmup_steps
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def save_checkpoint(model, optimizer, scheduler, step, loss, checkpoint_dir, model_size):
    """Save model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'step': step,
        'loss': loss
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_size}_step_{step}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    return checkpoint_path

def save_results(results, model_size, results_dir):
    """Save training results to JSON"""
    model_results_dir = os.path.join(results_dir, model_size)
    os.makedirs(model_results_dir, exist_ok=True)
    
    results_path = os.path.join(model_results_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results: {results_path}")
    return results_path

def train_step(model, batch, device):
    """Execute a single training step"""
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # Forward pass
    logits = model(input_ids, attention_mask)
    
    # Calculate loss (language modeling loss)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    # Calculate accuracy (for monitoring)
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        mask = labels != -100
        correct = (predictions == labels) & mask
        accuracy = correct.sum().float() / mask.sum().float()
    
    return loss, accuracy.item()

def evaluate_model(model, val_dataloader, device, max_eval_steps=None):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    num_steps = 0
    
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            if max_eval_steps and step >= max_eval_steps:
                break
                
            loss, accuracy = train_step(model, batch, device)
            total_loss += loss.item()
            total_accuracy += accuracy
            num_steps += 1
    
    model.train()
    return total_loss / num_steps, total_accuracy / num_steps

def run_pretraining(model_size, model_config_path, data_config_path, training_config_path, 
                   results_dir, logs_dir, checkpoints_dir):
    """
    Main pretraining function
    
    Args:
        model_size: Size of model to train
        model_config_path: Path to model configuration
        data_config_path: Path to data configuration  
        training_config_path: Path to training configuration
        results_dir: Directory to save results
        logs_dir: Directory for logs
        checkpoints_dir: Directory for checkpoints
        
    Returns:
        bool: True if training succeeded, False otherwise
    """
    
    print(f"Starting pretraining for {model_size} model")
    
    try:
        # Load configurations
        model_configs, data_config, training_config = load_configs(
            model_config_path, data_config_path, training_config_path
        )
        
        if model_size not in model_configs['models']:
            raise ValueError(f"Model size '{model_size}' not found in config")
        
        model_config = model_configs['models'][model_size]
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Setup logging
        use_wandb = setup_logging(model_size, training_config, logs_dir)
        
        # Create model
        print("Creating model...")
        model, _ = create_model(model_size, model_config_path)
        model = model.to(device)
        
        print(f"Model parameters: {model.get_num_params():,}")
        
        # Create datasets
        print("Loading datasets...")
        # First ensure full dataset is preprocessed
        full_train_dataset, full_val_dataset, _ = create_full_dataset(data_config_path)
        
        # Then sample according to model requirements
        train_dataset = create_sampled_dataset(full_train_dataset, model_config)
        val_dataset = create_sampled_dataset(full_val_dataset, model_config)
        
        print(f"Training samples: {len(train_dataset):,}")
        print(f"Validation samples: {len(val_dataset):,}")
        
        # Log tokens we're training on (from sampled dataset)
        if hasattr(train_dataset, 'get_token_stats'):
            train_stats = train_dataset.get_token_stats()
            print(f"Training on: {train_stats['actual_tokens']:,} tokens ({train_stats['actual_tokens']/1e9:.3f}B)")
        
        # Create data loaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=model_config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=model_config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Setup training
        max_steps = int(model_config['max_steps'])
        optimizer, scheduler, warmup_steps = create_optimizer_and_scheduler(
            model, model_config, max_steps
        )
        
        # Training loop
        print(f"Starting training for {max_steps:,} steps...")
        model.train()
        
        step = 0
        running_loss = 0
        best_val_loss = float('inf')
        
        # Timing
        training_start_time = time.time()
        step_times = []
        
        # Training metrics to collect
        training_metrics = {
            'model_size': model_size,
            'model_params': model.get_num_params(),
            'start_time': datetime.now().isoformat(),
            'config': model_config,
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'steps': []
        }
        
        # Add training token count
        if hasattr(train_dataset, 'get_token_stats'):
            training_metrics['training_tokens'] = train_dataset.get_token_stats()['actual_tokens']
        
        # Create progress bar for entire training
        progress_bar = tqdm(total=max_steps, desc=f"Training {model_size}", unit="step")
        
        for epoch in range(1000):  # Large number, will break by steps
            for batch in train_dataloader:
                if step >= max_steps:
                    break
                
                step_start_time = time.time()
                
                # Warmup learning rate
                if step < warmup_steps:
                    warmup_lr(optimizer, step, warmup_steps, model_config['learning_rate'])
                
                # Training step
                optimizer.zero_grad()
                loss, accuracy = train_step(model, batch, device)
                loss.backward()
                
                # Gradient clipping
                max_grad_norm = float(model_config.get('max_grad_norm', 1.0))
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                if scheduler and step >= warmup_steps:
                    scheduler.step()
                
                running_loss += loss.item()
                step += 1
                
                # Track step time
                step_time = time.time() - step_start_time
                step_times.append(step_time)
                
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{accuracy:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                # Logging
                log_every = int(model_config.get('log_every', 100))
                if step % log_every == 0:
                    avg_loss = running_loss / log_every
                    
                    log_data = {
                        'step': step,
                        'train_loss': avg_loss,
                        'train_accuracy': accuracy,
                        'learning_rate': optimizer.param_groups[0]['lr']
                    }
                    
                    if use_wandb:
                        wandb.log(log_data)
                    
                    # Don't print during training - tqdm progress bar shows this
                    running_loss = 0
                
                # Evaluation
                eval_every = int(model_config.get('eval_every', 1000))
                if step % eval_every == 0:
                    progress_bar.write("Running evaluation...")
                    val_loss, val_accuracy = evaluate_model(model, val_dataloader, device, max_eval_steps=100)
                    
                    eval_data = {
                        'step': step,
                        'val_loss': val_loss,
                        'val_accuracy': val_accuracy
                    }
                    
                    if use_wandb:
                        wandb.log(eval_data)
                    
                    progress_bar.write(f"Validation: Loss={val_loss:.4f}, Acc={val_accuracy:.4f}")
                    
                    # Save metrics
                    training_metrics['steps'].append(step)
                    training_metrics['train_losses'].append(avg_loss)
                    training_metrics['val_losses'].append(val_loss)
                    training_metrics['train_accuracies'].append(accuracy)
                    training_metrics['val_accuracies'].append(val_accuracy)
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(model, optimizer, scheduler, step, val_loss, 
                                      checkpoints_dir, f"{model_size}_best")
                
                # Regular checkpoint saving
                save_every = int(model_config.get('save_every', 5000))
                if step % save_every == 0:
                    save_checkpoint(model, optimizer, scheduler, step, loss.item(), 
                                  checkpoints_dir, model_size)
            
            if step >= max_steps:
                break
        
        progress_bar.close()
        
        # Calculate training time and timing statistics
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        avg_step_time = sum(step_times) / len(step_times) if step_times else 0
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED FOR {model_size.upper()}")
        print(f"{'='*60}")
        print(f"Total training time: {total_training_time/3600:.2f} hours ({total_training_time/60:.1f} minutes)")
        print(f"Average time per step: {avg_step_time:.3f} seconds")
        print(f"Steps per second: {1/avg_step_time:.2f}")
        print(f"Total steps completed: {step:,}")
        
        # Final evaluation and save
        print("Running final evaluation...")
        final_val_loss, final_val_accuracy = evaluate_model(model, val_dataloader, device)
        
        training_metrics['end_time'] = datetime.now().isoformat()
        training_metrics['total_training_time_seconds'] = total_training_time
        training_metrics['total_training_time_hours'] = total_training_time / 3600
        training_metrics['average_step_time_seconds'] = avg_step_time
        training_metrics['steps_per_second'] = 1 / avg_step_time if avg_step_time > 0 else 0
        training_metrics['final_val_loss'] = final_val_loss
        training_metrics['final_val_accuracy'] = final_val_accuracy
        training_metrics['total_steps'] = step
        
        # Save final results
        save_results(training_metrics, model_size, results_dir)
        
        # Save final checkpoint
        final_checkpoint = save_checkpoint(model, optimizer, scheduler, step, final_val_loss, 
                                         checkpoints_dir, f"{model_size}_final")
        
        if use_wandb:
            wandb.log({
                'final_val_loss': final_val_loss,
                'final_val_accuracy': final_val_accuracy,
                'total_steps': step,
                'total_training_time_hours': total_training_time / 3600,
                'average_step_time_seconds': avg_step_time,
                'steps_per_second': 1 / avg_step_time if avg_step_time > 0 else 0
            })
            wandb.finish()
        
        print(f" Pretraining completed for {model_size}")
        print(f"Final validation loss: {final_val_loss:.4f}")
        print(f"Final validation accuracy: {final_val_accuracy:.4f}")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"L Error during pretraining for {model_size}: {e}")
        import traceback
        traceback.print_exc()
        
        if use_wandb:
            wandb.finish()
        
        return False