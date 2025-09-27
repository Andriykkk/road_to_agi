"""
Pretraining module for language modeling (next token prediction)
"""

import os
import json
import time
import math
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

def setup_logging(model_size, model_config, training_config, logs_dir):
    """Setup logging with wandb"""
    if training_config.get('use_wandb', True):
        run_name = f"{model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create comprehensive config for wandb
        wandb_config = {
            'model_size': model_size,
            'model_params': model_config.get('params', 0),
            'target_tokens_billions': model_config.get('target_tokens_billions'),
            'max_steps': model_config.get('max_steps'),
            'learning_rate': model_config.get('learning_rate'),
            'batch_size': model_config.get('batch_size'),
            'gradient_accumulation_steps': model_config.get('gradient_accumulation_steps', 1),
            'effective_batch_size': model_config.get('batch_size', 32) * model_config.get('gradient_accumulation_steps', 1),
            'max_seq_len': model_config['architecture']['max_seq_len'],
            'n_layers': model_config['architecture']['n_layers'],
            'n_heads': model_config['architecture']['n_heads'],
            'd_model': model_config['architecture']['d_model'],
            'warmup_percent': model_config.get('warmup_percent'),
            'warmup_steps': model_config.get('warmup_steps'),
            'lr_scheduler': model_config.get('lr_scheduler'),
            'optimizer': model_config.get('optimizer'),
            'weight_decay': model_config.get('weight_decay')
        }
        
        wandb.init(
            project=training_config.get('wandb_project', 'scaling-laws'),
            entity=training_config.get('wandb_entity'),
            name=run_name,
            tags=[model_size, 'pretraining'],
            config=wandb_config,
            dir=logs_dir
        )
        return True
    return False

def create_optimizer_and_scheduler(model, model_config, total_steps):
    """Create optimizer and learning rate scheduler"""
    base_lr = float(model_config['learning_rate'])
    warmup_start_lr_ratio = model_config.get('warmup_start_lr_ratio', 0.1)
    
    # Start optimizer with warmup start LR
    optimizer = AdamW(
        model.parameters(),
        lr=base_lr * warmup_start_lr_ratio,
        weight_decay=float(model_config.get('weight_decay', 0.1))
    )
    
    scheduler_type = model_config.get('lr_scheduler', 'cosine')
    warmup_steps = int(model_config.get('warmup_steps', 1000))
    min_lr_ratio = model_config.get('min_lr_ratio', 0.1)
    
    if scheduler_type == 'cosine':
        # Cosine annealing from peak LR to min LR after warmup
        min_lr = base_lr * min_lr_ratio
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr)
    else:
        scheduler = None
        
    return optimizer, scheduler, warmup_steps

def warmup_lr(optimizer, step, warmup_steps, base_lr, warmup_start_lr_ratio=0.1):
    """Apply learning rate warmup from start_ratio to peak LR"""
    if step < warmup_steps:
        start_lr = float(base_lr) * warmup_start_lr_ratio
        target_lr = float(base_lr)
        progress = (step + 1) / warmup_steps
        lr = start_lr + (target_lr - start_lr) * progress
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
        
        # Calculate training steps from target tokens and batch size
        target_tokens = int(model_config['target_tokens_billions'] * 1e9)
        max_seq_len = model_config['architecture']['max_seq_len']
        batch_size = model_config['batch_size']
        gradient_accumulation_steps = model_config.get('gradient_accumulation_steps', 1)
        
        tokens_per_step = max_seq_len * batch_size * gradient_accumulation_steps
        calculated_steps = math.ceil(target_tokens / tokens_per_step)
        
        print(f"🧮 Calculated training steps: {calculated_steps:,}")
        print(f"   Target tokens: {target_tokens:,} ({model_config['target_tokens_billions']}B)")
        print(f"   Tokens per step: {tokens_per_step:,} (seq_len={max_seq_len} × batch={batch_size} × accum={gradient_accumulation_steps})")
        
        model_config['max_steps'] = calculated_steps
        
        # Calculate warmup steps from percentage
        warmup_percent = model_config.get('warmup_percent', 10)
        warmup_steps = int(calculated_steps * warmup_percent / 100)
        model_config['warmup_steps'] = warmup_steps
        
        print(f"   Warmup steps: {warmup_steps:,} ({warmup_percent}% of training)")
        
        # Calculate learning rate schedule parameters
        base_lr = float(model_config['learning_rate'])
        warmup_start_lr_ratio = model_config.get('warmup_start_lr_ratio', 0.1)
        min_lr_ratio = model_config.get('min_lr_ratio', 0.1)
        
        warmup_start_lr = base_lr * warmup_start_lr_ratio
        min_lr = base_lr * min_lr_ratio
        
        print(f"   LR schedule: {warmup_start_lr:.2e} → {base_lr:.2e} → {min_lr:.2e}")
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Setup logging
        use_wandb = setup_logging(model_size, model_config, training_config, logs_dir)
        
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
        best_val_loss = float('inf')
        gradient_accumulation_steps = model_config.get('gradient_accumulation_steps', 1)
        
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
        
        # Gradient accumulation tracking
        accumulation_step = 0
        accumulated_loss = 0
        step_start_time = None
        
        # Training loop - iterate through data until we reach max_steps
        data_iter = iter(train_dataloader)
        
        while step < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                # Reset dataloader when we reach the end
                data_iter = iter(train_dataloader)
                batch = next(data_iter)
            
            # Start timing when we begin a new optimization step
            if accumulation_step == 0:
                step_start_time = time.time()
            
            # Training forward pass
            loss, accuracy = train_step(model, batch, device)
            
            # Scale loss by accumulation steps
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            accumulated_loss += loss.item()
            accumulation_step += 1
            
            # Perform optimization step when accumulation is complete
            if accumulation_step == gradient_accumulation_steps:
                # Warmup learning rate
                if step < warmup_steps:
                    warmup_lr(optimizer, step, warmup_steps, 
                             model_config['learning_rate'], 
                             model_config.get('warmup_start_lr_ratio', 0.1))
                
                # Gradient clipping
                max_grad_norm = float(model_config.get('max_grad_norm', 1.0))
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Optimization step
                optimizer.step()
                optimizer.zero_grad()
                
                if scheduler and step >= warmup_steps:
                    scheduler.step()
                
                step += 1
                
                # Track step time (only for actual optimization steps)
                if step_start_time is not None:
                    step_time = time.time() - step_start_time
                    step_times.append(step_time)
                    steps_per_second = 1.0 / step_time if step_time > 0 else 0
                else:
                    step_time = 0
                    steps_per_second = 0
                
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'loss': f'{accumulated_loss:.4f}',
                    'acc': f'{accuracy:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                    'step/s': f'{steps_per_second:.2f}'
                })
                
                # Log training metrics every step to wandb
                if use_wandb:
                    wandb.log({
                        'train_loss': accumulated_loss,
                        'train_accuracy': accuracy,
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'step_time_seconds': step_time,
                        'steps_per_second': steps_per_second
                    }, step=step)
                
                # Run evaluation if needed (less frequent)
                eval_every = int(model_config.get('eval_every', 1000))
                if step % eval_every == 0:
                    progress_bar.write("Running evaluation...")
                    val_loss, val_accuracy = evaluate_model(model, val_dataloader, device, max_eval_steps=100)
                    
                    # Log validation metrics to wandb
                    if use_wandb:
                        wandb.log({
                            'val_loss': val_loss,
                            'val_accuracy': val_accuracy
                        }, step=step)
                    
                    progress_bar.write(f"Validation: Loss={val_loss:.4f}, Acc={val_accuracy:.4f}")
                    
                    # Save metrics for final results
                    training_metrics['steps'].append(step)
                    training_metrics['train_losses'].append(accumulated_loss)
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
                    save_checkpoint(model, optimizer, scheduler, step, accumulated_loss, 
                                  checkpoints_dir, model_size)
                
                # Reset accumulation
                accumulation_step = 0
                accumulated_loss = 0
        
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