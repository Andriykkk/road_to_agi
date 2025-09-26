import torch
from torch.utils.data import Dataset
import math
import random
import numpy as np

class DistributedSampler:
    """Simple sampler that distributes data across the full dataset using skip sampling"""
    
    def __init__(self, dataset_size, target_tokens_billions, max_length, seed=42):
        """
        Args:
            dataset_size: Total size of the dataset
            target_tokens_billions: How many billion tokens we want to sample
            max_length: Maximum sequence length (to estimate tokens per sample)
            seed: Random seed for reproducibility
        """
        self.dataset_size = dataset_size
        self.seed = seed
        
        # Estimate tokens per sample and calculate needed samples
        avg_tokens_per_sample = max_length * 0.8  # Assume 80% of max length
        target_samples = int((target_tokens_billions * 1e9) / avg_tokens_per_sample)
        
        # Simple logic:
        if target_samples >= dataset_size:
            # Need all data - use everything
            self.skip_interval = 1
            self.num_samples = dataset_size
        else:
            # Need less data - calculate skip interval
            self.skip_interval = dataset_size // target_samples
            self.num_samples = target_samples
        
        print(f"Dataset size: {dataset_size:,}, Target samples: {target_samples:,}")
        print(f"Using {self.num_samples:,} samples with skip interval: {self.skip_interval}")
        
        # Generate distributed indices
        self.indices = self._generate_distributed_indices()
        
    def _generate_distributed_indices(self):
        """Generate indices distributed across the full dataset"""
        if self.skip_interval == 1:
            # Use all samples
            return list(range(self.dataset_size))
        else:
            # Skip sampling - take every Nth sample
            indices = list(range(0, self.dataset_size, self.skip_interval))
            return indices[:self.num_samples]  # Trim to exact number needed
        
    def __iter__(self):
        # Shuffle indices while maintaining distribution
        random.seed(self.seed)
        shuffled_indices = self.indices.copy()
        random.shuffle(shuffled_indices)
        return iter(shuffled_indices)
        
    def __len__(self):
        return len(self.indices)
    
    def get_indices(self):
        """Get the sampling indices"""
        return self.indices.copy()

class SampledDataset(Dataset):
    """Dataset wrapper that samples distributed chunks across the token sequence"""
    
    def __init__(self, original_dataset, target_tokens_billions, max_length, seed=42):
        """
        Args:
            original_dataset: Original dataset (PreprocessedC4Dataset with raw tokens)
            target_tokens_billions: How many billion tokens we want to sample
            max_length: Maximum sequence length for chunking
            seed: Random seed
        """
        self.original_dataset = original_dataset
        self.target_tokens_billions = target_tokens_billions
        self.max_length = max_length
        self.seed = seed
        
        # Get target token count
        target_tokens = int(target_tokens_billions * 1e9)
        
        # Get dataset stats
        if hasattr(original_dataset, 'get_token_stats'):
            stats = original_dataset.get_token_stats()
            available_tokens = stats['actual_tokens']
        else:
            available_tokens = len(original_dataset)
        
        # Calculate how many chunks we need
        target_chunks = target_tokens // max_length
        max_possible_chunks = available_tokens // max_length
        
        if target_chunks >= max_possible_chunks:
            # Use all possible chunks
            self.chunk_indices = list(range(max_possible_chunks))
            actual_tokens = max_possible_chunks * max_length
        else:
            # Sample distributed chunks across the dataset
            # Calculate skip interval to distribute chunks evenly
            skip_interval = max_possible_chunks // target_chunks
            self.chunk_indices = list(range(0, max_possible_chunks, max(1, skip_interval)))[:target_chunks]
            actual_tokens = len(self.chunk_indices) * max_length
        
        print(f"Sampled {len(self.chunk_indices):,} chunks ({actual_tokens:,} tokens, {actual_tokens/1e9:.3f}B)")
        print(f"Target was {target_chunks:,} chunks ({target_tokens:,} tokens, {target_tokens_billions:.3f}B)")
        print(f"Chunk indices range: {min(self.chunk_indices) if self.chunk_indices else 0} to {max(self.chunk_indices) if self.chunk_indices else 0}")
        print(f"Skip interval: {skip_interval if target_chunks < max_possible_chunks else 1}")
        
    def __len__(self):
        # Return number of sampled chunks
        return len(self.chunk_indices)
        
    def __getitem__(self, idx):
        # Get the chunk index for this sample
        chunk_idx = self.chunk_indices[idx]
        
        # Calculate start position in the original token sequence
        start_idx = chunk_idx * self.max_length
        end_idx = start_idx + self.max_length
        
        # Get token sequence from original dataset (efficiently handle chunked datasets)
        if hasattr(self.original_dataset, 'get_token_slice'):
            # Use efficient slice method for chunked datasets
            input_ids = self.original_dataset.get_token_slice(start_idx, end_idx)
        else:
            # Fallback for legacy datasets
            input_ids = self.original_dataset.tokens[start_idx:end_idx]
        
        # Convert to numpy if it's a list (for legacy compatibility)
        if isinstance(input_ids, list):
            input_ids = np.array(input_ids, dtype=np.uint16)
        
        # Pad if necessary (shouldn't happen with proper chunking)
        if len(input_ids) < self.max_length:
            # Pad with tokenizer's pad token (0 for GPT-2)
            pad_length = self.max_length - len(input_ids)
            padding = np.zeros(pad_length, dtype=np.uint16)
            input_ids = np.concatenate([input_ids, padding])
            attention_mask = np.concatenate([
                np.ones(len(input_ids) - pad_length, dtype=np.uint16),
                np.zeros(pad_length, dtype=np.uint16)
            ])
        else:
            attention_mask = np.ones(self.max_length, dtype=np.uint16)
        
        # Create labels (same as input_ids for language modeling)
        labels = input_ids.copy()
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def get_token_stats(self):
        """Get actual token statistics for this sampled dataset"""
        return {
            'actual_tokens': len(self.chunk_indices) * self.max_length,
            'sequences': len(self.chunk_indices),
            'max_length': self.max_length,
            'target_tokens_billions': self.target_tokens_billions
        }

def create_sampled_dataset(dataset, model_config, seed=42):
    """Create a sampled dataset based on model configuration
    
    Args:
        dataset: Original dataset
        model_config: Model configuration containing target tokens and architecture
        seed: Random seed
        
    Returns:
        SampledDataset instance or original dataset
    """
    target_tokens_billions = model_config.get('target_tokens_billions', 1.0)
    max_length = model_config['architecture']['max_seq_len']
    
    # Always create sampled dataset - it will handle full dataset case internally
    return SampledDataset(dataset, target_tokens_billions, max_length, seed)