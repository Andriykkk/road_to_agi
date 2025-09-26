import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import yaml
import os
import pickle
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm
import warnings
import os
import math
import glob

# Suppress tokenizer warnings about long sequences
warnings.filterwarnings("ignore", message="Token indices sequence length is longer than the specified maximum sequence length")
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def tokenize_batch(args):
    """Tokenize batch and concatenate with EOS tokens"""
    import warnings
    import os
    import numpy as np
    warnings.filterwarnings("ignore")
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    texts, tokenizer_name = args
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize all texts and join with EOS - use lists first, then convert to numpy
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False, max_length=10000000000)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_token_id)  # Add EOS between documents
    
    # Convert to numpy array for memory efficiency - use uint16 since vocab size is ~50k
    return np.array(all_tokens, dtype=np.uint16)

class ChunkedC4Dataset(Dataset):
    """Memory-efficient C4 dataset that stores data in chunks and loads on-demand"""
    
    def __init__(self, split='train', max_length=1024, tokenizer_name='gpt2', 
                 target_tokens_billions=1.0, cache_dir='data_cache', force_preprocess=False,
                 max_file_size_gb=1.0, chunk_size_tokens=50_000_000):
        """
        Args:
            split: Dataset split ('train', 'validation')
            max_length: Maximum sequence length
            tokenizer_name: Tokenizer to use
            target_tokens_billions: How many billion tokens to preprocess and cache
            cache_dir: Directory to cache preprocessed data
            force_preprocess: Whether to force reprocessing even if cache exists
            max_file_size_gb: Maximum size per chunk file in GB
            chunk_size_tokens: Number of tokens per chunk
        """
        self.split = split
        self.max_length = max_length
        self.tokenizer_name = tokenizer_name
        self.target_tokens_billions = target_tokens_billions
        self.cache_dir = cache_dir
        self.max_file_size_gb = max_file_size_gb
        self.chunk_size_tokens = chunk_size_tokens
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Calculate optimal chunk size based on file size limit
        # uint16 = 2 bytes per token
        max_tokens_per_file = int((max_file_size_gb * 1024**3) / 2)
        self.actual_chunk_size = min(chunk_size_tokens, max_tokens_per_file)
        
        # Cache file pattern
        self.cache_prefix = f"c4_{split}_{tokenizer_name.replace('/', '_')}_{max_length}_{target_tokens_billions}B"
        
        # Initialize chunk management
        self.chunk_files = []
        self.chunk_sizes = []  # Track actual tokens in each chunk
        self.total_tokens = 0
        self.current_chunk = None
        self.current_chunk_idx = -1
        
        # Check if chunked cache exists and is sufficient
        need_reprocess = True
        if not force_preprocess:
            self._discover_existing_chunks()
            needed_tokens = int(target_tokens_billions * 1e9)
            
            if self.total_tokens >= needed_tokens:
                print(f"✅ Found sufficient cached chunks")
                print(f"Cached: {self.total_tokens:,} tokens ({self.total_tokens/1e9:.3f}B) in {len(self.chunk_files)} files")
                print(f"Needed: {needed_tokens:,} tokens ({target_tokens_billions:.3f}B)")
                need_reprocess = False
            else:
                print(f"⚠️  Insufficient cached data. Need {target_tokens_billions:.3f}B, have {self.total_tokens/1e9:.3f}B")
                print("Will preprocess additional data...")
                self._cleanup_old_chunks()
        
        if need_reprocess:
            print(f"Preprocessing C4 data with {target_tokens_billions}B tokens...")
            print(f"Using chunks of {self.actual_chunk_size:,} tokens (max {max_file_size_gb}GB per file)")
            print("This may take a while... The dataset will be cached in chunks for future use.")
            
            self._preprocess_data_chunked()
            print(f"✅ Data preprocessing completed and cached in {len(self.chunk_files)} chunks")
    
    def _discover_existing_chunks(self):
        """Discover existing chunk files and load metadata"""
        pattern = os.path.join(self.cache_dir, f"{self.cache_prefix}_chunk_*.npz")
        chunk_files = sorted(glob.glob(pattern))
        
        self.chunk_files = []
        self.chunk_sizes = []
        self.total_tokens = 0
        
        for chunk_file in chunk_files:
            try:
                # Load chunk metadata without loading the data
                with np.load(chunk_file) as data:
                    chunk_size = data['tokens'].shape[0]
                    self.chunk_files.append(chunk_file)
                    self.chunk_sizes.append(chunk_size)
                    self.total_tokens += chunk_size
            except:
                print(f"⚠️  Error reading chunk {chunk_file}, skipping")
                continue
    
    def _cleanup_old_chunks(self):
        """Remove old chunk files"""
        pattern = os.path.join(self.cache_dir, f"{self.cache_prefix}_chunk_*.npz")
        for file in glob.glob(pattern):
            try:
                os.remove(file)
            except:
                pass
        self.chunk_files = []
        self.chunk_sizes = []
        self.total_tokens = 0
    
    def _save_chunk(self, tokens, chunk_idx):
        """Save a chunk of tokens to disk"""
        chunk_file = os.path.join(self.cache_dir, f"{self.cache_prefix}_chunk_{chunk_idx:04d}.npz")
        np.savez_compressed(chunk_file, tokens=tokens)
        self.chunk_files.append(chunk_file)
        self.chunk_sizes.append(len(tokens))
        print(f"Saved chunk {chunk_idx} with {len(tokens):,} tokens to {os.path.basename(chunk_file)}")
    
    def _load_chunk(self, chunk_idx):
        """Load a specific chunk into memory"""
        if self.current_chunk_idx == chunk_idx and self.current_chunk is not None:
            return self.current_chunk
        
        if chunk_idx >= len(self.chunk_files):
            raise IndexError(f"Chunk {chunk_idx} does not exist")
        
        # Load new chunk
        with np.load(self.chunk_files[chunk_idx]) as data:
            self.current_chunk = data['tokens']
            self.current_chunk_idx = chunk_idx
            return self.current_chunk
    
    def _preprocess_data_chunked(self):
        """Preprocess and tokenize the data in chunks"""
        # Load tokenizer to estimate tokens per sample
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load C4 dataset
        print("Loading C4 dataset...")
        try:
            dataset = load_dataset('allenai/c4', 'en', split=self.split, streaming=True, trust_remote_code=True)
        except Exception as e:
            print(f"Failed to load allenai/c4: {e}")
            try:
                dataset = load_dataset('c4', 'en', split=self.split, streaming=True, trust_remote_code=True)
            except Exception as e2:
                print(f"Failed to load c4 dataset: {e2}")
                print("Using mc4 (multilingual C4) as fallback...")
                dataset = load_dataset('mc4', 'en', split=self.split, streaming=True, trust_remote_code=True)
        
        # Target tokens and processing settings
        target_tokens = int(self.target_tokens_billions * 1e9)
        batch_size = 1000
        num_processes = min(cpu_count(), 32)
        
        # Load processing config if available
        try:
            with open('config/data.yaml', 'r') as f:
                data_config = yaml.safe_load(f)
            num_processes = min(data_config.get('max_processes', num_processes), 32)
            batch_size = data_config.get('batch_size', batch_size)
        except:
            pass
        
        print(f"Target: {target_tokens:,} tokens, chunk size: {self.actual_chunk_size:,}")
        print(f"Using {num_processes} processes, batch size {batch_size}")
        
        # Process data in chunks
        current_chunk_tokens = []
        chunk_idx = 0
        current_batch = []
        total_processed = 0
        
        pbar = tqdm(total=target_tokens, desc="Processing", unit="tokens", unit_scale=True)
        
        with Pool(num_processes) as pool:
            for sample in dataset:
                current_batch.append(sample['text'])
                
                if len(current_batch) >= batch_size:
                    # Process batch
                    batch_tokens = tokenize_batch((current_batch, self.tokenizer_name))
                    current_batch = []
                    
                    # Add to current chunk
                    if len(current_chunk_tokens) == 0:
                        current_chunk_tokens = batch_tokens
                    else:
                        current_chunk_tokens = np.concatenate([current_chunk_tokens, batch_tokens])
                    
                    # Save chunk if it's large enough
                    while len(current_chunk_tokens) >= self.actual_chunk_size:
                        chunk_data = current_chunk_tokens[:self.actual_chunk_size]
                        remaining_data = current_chunk_tokens[self.actual_chunk_size:]
                        
                        self._save_chunk(chunk_data, chunk_idx)
                        chunk_idx += 1
                        
                        current_chunk_tokens = remaining_data
                        total_processed += len(chunk_data)
                        pbar.update(len(chunk_data))
                        
                        if total_processed >= target_tokens:
                            break
                    
                    if total_processed >= target_tokens:
                        break
            
            # Process remaining batch
            if current_batch and total_processed < target_tokens:
                batch_tokens = tokenize_batch((current_batch, self.tokenizer_name))
                if len(current_chunk_tokens) == 0:
                    current_chunk_tokens = batch_tokens
                else:
                    current_chunk_tokens = np.concatenate([current_chunk_tokens, batch_tokens])
        
        # Save final chunk if it has data
        if len(current_chunk_tokens) > 0:
            # Trim to target if needed
            if total_processed + len(current_chunk_tokens) > target_tokens:
                remaining_needed = target_tokens - total_processed
                current_chunk_tokens = current_chunk_tokens[:remaining_needed]
            
            self._save_chunk(current_chunk_tokens, chunk_idx)
            total_processed += len(current_chunk_tokens)
            pbar.update(len(current_chunk_tokens))
        
        pbar.close()
        self.total_tokens = total_processed
        print(f"Processed {total_processed:,} tokens ({total_processed/1e9:.3f}B) in {len(self.chunk_files)} chunks")
    
    def _preprocess_data(self):
        """Preprocess and tokenize the data using multiprocessing"""
        # Load tokenizer to estimate tokens per sample
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load C4 dataset using the modern approach
        print("Loading C4 dataset...")
        try:
            # Try the modern allenai/c4 dataset
            dataset = load_dataset('allenai/c4', 'en', split=self.split, streaming=True, trust_remote_code=True)
        except Exception as e:
            print(f"Failed to load allenai/c4: {e}")
            # Fallback to alternative
            try:
                dataset = load_dataset('c4', 'en', split=self.split, streaming=True, trust_remote_code=True)
            except Exception as e2:
                print(f"Failed to load c4 dataset: {e2}")
                # Use a smaller alternative for testing
                print("Using mc4 (multilingual C4) as fallback...")
                dataset = load_dataset('mc4', 'en', split=self.split, streaming=True, trust_remote_code=True)
        
        # Stream processing with target token count
        target_tokens = int(self.target_tokens_billions * 1e9)
        print(f"Target: {target_tokens:,} tokens ({self.target_tokens_billions:.3f}B)")
        
        # Multiprocessing settings
        batch_size = 1000
        num_processes = min(cpu_count(), 32)
        
        # Check data config for multiprocessing settings
        try:
            import yaml
            with open('config/data.yaml', 'r') as f:
                data_config = yaml.safe_load(f)
            if data_config.get('max_processes') is not None:
                num_processes = min(data_config['max_processes'], 32)
            if data_config.get('batch_size') is not None:
                batch_size = data_config['batch_size']
        except:
            pass
        
        print(f"Using {num_processes} processes, batch size {batch_size}")
        
        # Stream and tokenize with progress tracking
        all_tokens = []
        current_batch = []
        
        # Progress bar for token collection
        pbar = tqdm(total=target_tokens, desc="Tokenizing", unit="tokens", unit_scale=True)
        
        with Pool(num_processes) as pool:
            for sample in dataset:
                current_batch.append(sample['text'])
                
                # Process when batch full
                if len(current_batch) >= batch_size:
                    batch_tokens = tokenize_batch((current_batch, self.tokenizer_name))
                    all_tokens.extend(batch_tokens)
                    current_batch = []
                    
                    # Update progress
                    pbar.update(len(batch_tokens))
                    
                    # Stop when target reached
                    if len(all_tokens) >= target_tokens:
                        break
            
            # Process remaining
            if current_batch and len(all_tokens) < target_tokens:
                batch_tokens = tokenize_batch((current_batch, self.tokenizer_name))
                all_tokens.extend(batch_tokens)
                pbar.update(len(batch_tokens))
        
        pbar.close()
        
        # Trim to target size
        all_tokens = all_tokens[:target_tokens]
        
        print(f"Collected {len(all_tokens):,} tokens ({len(all_tokens)/1e9:.3f}B)")
        
        # Store as single sequence
        self.tokens = all_tokens
        self.actual_tokens_count = len(all_tokens)
        
        return all_tokens
    
    def __len__(self):
        return self.total_tokens
        
    def __getitem__(self, idx):
        """Get a token at global index by loading appropriate chunk"""
        if idx >= self.total_tokens:
            raise IndexError(f"Index {idx} out of range (total tokens: {self.total_tokens})")
        
        # Find which chunk contains this index
        current_pos = 0
        for chunk_idx, chunk_size in enumerate(self.chunk_sizes):
            if current_pos + chunk_size > idx:
                # Found the chunk, load it and return the token
                chunk = self._load_chunk(chunk_idx)
                local_idx = idx - current_pos
                return chunk[local_idx]
            current_pos += chunk_size
        
        raise IndexError(f"Could not locate index {idx}")
    
    def get_token_slice(self, start_idx, end_idx):
        """Get a slice of tokens efficiently across chunks"""
        if start_idx >= self.total_tokens:
            raise IndexError(f"Start index {start_idx} out of range")
        
        end_idx = min(end_idx, self.total_tokens)
        slice_length = end_idx - start_idx
        
        if slice_length <= 0:
            return np.array([], dtype=np.uint16)
        
        # Find starting chunk
        current_pos = 0
        start_chunk_idx = 0
        for chunk_idx, chunk_size in enumerate(self.chunk_sizes):
            if current_pos + chunk_size > start_idx:
                start_chunk_idx = chunk_idx
                break
            current_pos += chunk_size
        
        # Collect tokens across chunks
        result_tokens = []
        remaining = slice_length
        chunk_start_pos = current_pos
        
        for chunk_idx in range(start_chunk_idx, len(self.chunk_sizes)):
            if remaining <= 0:
                break
                
            chunk = self._load_chunk(chunk_idx)
            
            # Calculate slice within this chunk
            if chunk_idx == start_chunk_idx:
                local_start = start_idx - chunk_start_pos
            else:
                local_start = 0
                
            local_end = min(local_start + remaining, len(chunk))
            chunk_slice = chunk[local_start:local_end]
            
            result_tokens.append(chunk_slice)
            remaining -= len(chunk_slice)
            chunk_start_pos += self.chunk_sizes[chunk_idx]
        
        if len(result_tokens) == 1:
            return result_tokens[0]
        else:
            return np.concatenate(result_tokens)
    
    def get_token_stats(self):
        """Get actual token statistics"""
        return {
            'actual_tokens': self.total_tokens,
            'sequences': 1,  # One long sequence
            'max_length': self.total_tokens,
            'token_efficiency': 1.0,  # All tokens are real data
            'num_chunks': len(self.chunk_files),
            'chunk_sizes': self.chunk_sizes.copy()
        }

class PreprocessedC4Dataset(Dataset):
    """C4 dataset that preprocesses and caches tokenized data (legacy - loads all in memory)"""
    
    def __init__(self, split='train', max_length=1024, tokenizer_name='gpt2', 
                 target_tokens_billions=1.0, cache_dir='data_cache', force_preprocess=False):
        """
        Args:
            split: Dataset split ('train', 'validation')
            max_length: Maximum sequence length
            tokenizer_name: Tokenizer to use
            target_tokens_billions: How many billion tokens to preprocess and cache
            cache_dir: Directory to cache preprocessed data
            force_preprocess: Whether to force reprocessing even if cache exists
        """
        self.split = split
        self.max_length = max_length
        self.tokenizer_name = tokenizer_name
        self.target_tokens_billions = target_tokens_billions
        self.cache_dir = cache_dir
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache file path
        cache_filename = f"c4_{split}_{tokenizer_name.replace('/', '_')}_{max_length}_{target_tokens_billions}B.pkl"
        self.cache_path = os.path.join(cache_dir, cache_filename)
        
        # Check if cache exists and is large enough
        need_reprocess = True
        if os.path.exists(self.cache_path) and not force_preprocess:
            try:
                # Load existing cache to check size
                with open(self.cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check if cached dataset is large enough
                cached_tokens = len(cached_data) if isinstance(cached_data, list) else 0
                needed_tokens = target_tokens_billions * 1e9
                
                if cached_tokens >= needed_tokens:
                    print(f"✅ Found sufficient cached data at {self.cache_path}")
                    print(f"Cached: {cached_tokens:,} tokens ({cached_tokens/1e9:.3f}B)")
                    print(f"Needed: {needed_tokens:,} tokens ({target_tokens_billions:.3f}B)")
                    self.tokens = cached_data
                    need_reprocess = False
                else:
                    print(f"⚠️  Cached data too small. Need {target_tokens_billions:.3f}B, have {cached_tokens/1e9:.3f}B")
                    print("Removing old cache and reprocessing...")
                    os.remove(self.cache_path)
            except:
                print("⚠️  Error reading cache, will reprocess")
                if os.path.exists(self.cache_path):
                    os.remove(self.cache_path)
        
        if need_reprocess:
            print(f"Preprocessing C4 data with {target_tokens_billions}B tokens...")
            print("This may take a while... The dataset will be cached for future use.")
            
            self.tokens = self._preprocess_data()
            
            # Save to cache
            print(f"Saving preprocessed data to {self.cache_path}")
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.tokens, f)
            print(f"✅ Data preprocessing completed and cached")
    
    def _preprocess_data(self):
        """Preprocess and tokenize the data using multiprocessing"""
        # Load tokenizer to estimate tokens per sample
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load C4 dataset using the modern approach
        print("Loading C4 dataset...")
        try:
            # Try the modern allenai/c4 dataset
            dataset = load_dataset('allenai/c4', 'en', split=self.split, streaming=True, trust_remote_code=True)
        except Exception as e:
            print(f"Failed to load allenai/c4: {e}")
            # Fallback to alternative
            try:
                dataset = load_dataset('c4', 'en', split=self.split, streaming=True, trust_remote_code=True)
            except Exception as e2:
                print(f"Failed to load c4 dataset: {e2}")
                # Use a smaller alternative for testing
                print("Using mc4 (multilingual C4) as fallback...")
                dataset = load_dataset('mc4', 'en', split=self.split, streaming=True, trust_remote_code=True)
        
        # Stream processing with target token count
        target_tokens = int(self.target_tokens_billions * 1e9)
        print(f"Target: {target_tokens:,} tokens ({self.target_tokens_billions:.3f}B)")
        
        # Multiprocessing settings
        batch_size = 1000
        num_processes = min(cpu_count(), 32)
        
        # Check data config for multiprocessing settings
        try:
            import yaml
            with open('config/data.yaml', 'r') as f:
                data_config = yaml.safe_load(f)
            if data_config.get('max_processes') is not None:
                num_processes = min(data_config['max_processes'], 32)
            if data_config.get('batch_size') is not None:
                batch_size = data_config['batch_size']
        except:
            pass
        
        print(f"Using {num_processes} processes, batch size {batch_size}")
        
        # Stream and tokenize with progress tracking
        all_tokens = []
        current_batch = []
        
        # Progress bar for token collection
        pbar = tqdm(total=target_tokens, desc="Tokenizing", unit="tokens", unit_scale=True)
        
        with Pool(num_processes) as pool:
            for sample in dataset:
                current_batch.append(sample['text'])
                
                # Process when batch full
                if len(current_batch) >= batch_size:
                    batch_tokens = tokenize_batch((current_batch, self.tokenizer_name))
                    if len(all_tokens) == 0:
                        all_tokens = batch_tokens
                    else:
                        all_tokens = np.concatenate([all_tokens, batch_tokens])
                    current_batch = []
                    
                    # Update progress
                    pbar.update(len(batch_tokens))
                    
                    # Stop when target reached
                    if len(all_tokens) >= target_tokens:
                        break
            
            # Process remaining
            if current_batch and len(all_tokens) < target_tokens:
                batch_tokens = tokenize_batch((current_batch, self.tokenizer_name))
                if len(all_tokens) == 0:
                    all_tokens = batch_tokens
                else:
                    all_tokens = np.concatenate([all_tokens, batch_tokens])
                pbar.update(len(batch_tokens))
        
        pbar.close()
        
        # Trim to target size
        all_tokens = all_tokens[:target_tokens]
        
        print(f"Collected {len(all_tokens):,} tokens ({len(all_tokens)/1e9:.3f}B)")
        
        return all_tokens
    
    def __len__(self):
        return len(self.tokens)
        
    def __getitem__(self, idx):
        # This dataset just stores the raw token sequence
        # Chunking will be handled by the sampler
        return self.tokens[idx]
    
    def get_token_stats(self):
        """Get actual token statistics"""
        return {
            'actual_tokens': len(self.tokens),
            'sequences': 1,  # One long sequence
            'max_length': len(self.tokens),
            'token_efficiency': 1.0  # All tokens are real data
        }

class C4Dataset(Dataset):
    """Simple C4 dataset for small-scale testing - uses concatenation approach"""
    def __init__(self, split='train', max_length=1024, tokenizer_name='gpt2', max_samples=10000):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load limited dataset
        try:
            dataset = load_dataset('allenai/c4', 'en', split=split, streaming=True, trust_remote_code=True)
        except:
            try:
                dataset = load_dataset('c4', 'en', split=split, streaming=True, trust_remote_code=True)
            except:
                # Fallback to mc4
                dataset = load_dataset('mc4', 'en', split=split, streaming=True, trust_remote_code=True)
        
        # Collect texts
        texts = []
        print(f"Loading {max_samples:,} samples from C4...")
        for i, sample in enumerate(tqdm(dataset, total=max_samples)):
            if i >= max_samples:
                break
            texts.append(sample['text'])
        
        # Use the same concatenation approach as the preprocessing
        print("Tokenizing and concatenating texts...")
        self.data, actual_tokens = tokenize_batch((texts, tokenizer_name, max_length))
        print(f"Created {len(self.data)} sequences from {len(texts)} texts")
        print(f"Actual tokens: {actual_tokens:,} ({actual_tokens/1e9:.3f}B)")
        self.actual_tokens_count = actual_tokens
            
    def __len__(self):
        return len(self.data)
            
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_token_stats(self):
        """Get actual token statistics"""
        return {
            'actual_tokens': getattr(self, 'actual_tokens_count', 0),
            'sequences': len(self.data),
            'max_length': self.max_length,
            'token_efficiency': getattr(self, 'actual_tokens_count', 0) / (len(self.data) * self.max_length) if len(self.data) > 0 else 0
        }

def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    """Create DataLoader from dataset"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

def load_data_config(config_path='config/data.yaml'):
    """Load data configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_full_dataset(data_config_path='config/data.yaml'):
    """Create the full preprocessed dataset (once) based on data config
    
    Args:
        data_config_path: Path to data configuration file
        
    Returns:
        train_dataset, val_dataset, data_config
    """
    data_config = load_data_config(data_config_path)
    
    use_preprocessed = data_config.get('use_preprocessed', True)
    preprocess_tokens_billions = data_config.get('preprocess_tokens_billions', 300.0)
    max_sequence_length = data_config.get('max_sequence_length', 1024)
    tokenizer_name = data_config.get('tokenizer', 'gpt2')
    cache_dir = data_config.get('cache_dir', 'data_cache')
    
    if use_preprocessed:
        # Use chunked dataset by default for memory efficiency
        use_chunked = data_config.get('use_chunked', True)
        max_file_size_gb = data_config.get('max_file_size_gb', 1.0)
        
        if use_chunked:
            train_dataset = ChunkedC4Dataset(
                split=data_config.get('train_split', 'train'),
                max_length=max_sequence_length,
                tokenizer_name=tokenizer_name,
                target_tokens_billions=preprocess_tokens_billions,
                cache_dir=cache_dir,
                max_file_size_gb=max_file_size_gb
            )
            
            val_dataset = ChunkedC4Dataset(
                split=data_config.get('validation_split', 'validation'),
                max_length=max_sequence_length,
                tokenizer_name=tokenizer_name,
                target_tokens_billions=preprocess_tokens_billions * data_config.get('validation_ratio', 0.1),
                cache_dir=cache_dir,
                max_file_size_gb=max_file_size_gb
            )
        else:
            # Legacy in-memory dataset
            train_dataset = PreprocessedC4Dataset(
                split=data_config.get('train_split', 'train'),
                max_length=max_sequence_length,
                tokenizer_name=tokenizer_name,
                target_tokens_billions=preprocess_tokens_billions,
                cache_dir=cache_dir
            )
            
            val_dataset = PreprocessedC4Dataset(
                split=data_config.get('validation_split', 'validation'),
                max_length=max_sequence_length,
                tokenizer_name=tokenizer_name,
                target_tokens_billions=preprocess_tokens_billions * data_config.get('validation_ratio', 0.1),
                cache_dir=cache_dir
            )
    else:
        # Use simple dataset for testing
        max_samples = 50000  # Larger for full dataset
        train_dataset = C4Dataset(
            split=data_config.get('train_split', 'train'),
            max_length=max_sequence_length,
            tokenizer_name=tokenizer_name,
            max_samples=max_samples
        )
        
        val_dataset = C4Dataset(
            split=data_config.get('validation_split', 'validation'),
            max_length=max_sequence_length,
            tokenizer_name=tokenizer_name,
            max_samples=max_samples // 10
        )
    
    return train_dataset, val_dataset, data_config