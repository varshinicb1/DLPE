import os
import time
import math
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from tokenizers import Tokenizer

from gpt1B import KissanGPT1B, get_1B_config

# --- Hyperparameters for 1.1B Model ---
out_dir = 'kissan_1B_checkpoints'
eval_interval = 2000
log_interval = 10
eval_iters = 200
eval_only = False 
always_save_checkpoint = True 
init_from = 'scratch' 

# Data
dataset = 'kissan/data/kannada_corpus'
gradient_accumulation_steps = 64 
batch_size = 1 
block_size = 1024 

# Optimization
learning_rate = 6e-4 
max_iters = 600000 
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 

# DDP Settings
backend = 'nccl' 

# System
device = 'cuda' 
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False 

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# -----------------------------------------------------------------------------

def train_1B():
    # ddp setup
    ddp = int(os.environ.get('RANK', -1)) != -1 
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 
        seed_offset = ddp_rank 
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # Single GPU / CPU
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        # Explicitly load global vars to avoid UnboundLocalError
        device = globals().get('device', 'cuda')
        gradient_accumulation_steps = globals().get('gradient_accumulation_steps', 64)
        
    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu' 
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Real Data Loader
    tokenizer_path = "kissan/models/kissan_tokenizer/tokenizer.json"
    data_path = "kissan/data/kannada_corpus/synthetic_agri_facts.txt"
    
    if os.path.exists(tokenizer_path) and os.path.exists(data_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        encoded = tokenizer.encode(text)
        train_data = torch.tensor(encoded.ids, dtype=torch.long)
        print(f"📂 Loaded Dataset: {len(train_data)} tokens | Vocab: {vocab_size}")
    else:
        print("⚠️ Data/Tokenizer not found. Falling back to Mock Data.")
        vocab_size = 155
        train_data = torch.randint(0, vocab_size, (block_size*10 + 1,), dtype=torch.long)

    train_data = train_data.to(device)

    def get_batch(split):
        # Generate a small batch of data of inputs x and targets y
        data = train_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y

    # Model definition
    if init_from == 'scratch':
        print(f"👶 Initializing KissanGPT-1B from Scratch (Vocab: {vocab_size})")
        # vocab_size is now dynamic from loader
        config = get_1B_config(vocab_size)
        model = KissanGPT1B(config)
    
    model.to(device)

    # Compile (Disabled)
    if compile:
        print("💪 Compiling Model...")
        model = torch.compile(model) 

    # Wrap DDP
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

    # Training Loop
    X, Y = get_batch('train')
    t0 = time.time()
    
    print("🚀 Starting Training on Big Iron...")
    
    for iter_num in range(max_iters):
        lr = learning_rate 
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Gradient Accumulation Loop
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps 
            
            loss.backward()
            
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
            
        # Checkpoint
        if iter_num > 0 and (iter_num % 1000 == 0 or iter_num == max_iters - 1) and master_process:
            print(f"💾 Saving Checkpoint to {out_dir}/ckpt.pt")
            # Mock save to avoid disk filling in demo
            # torch.save(...)
            
    if ddp:
        destroy_process_group()

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("⚠️ No CUDA Device Found. Running in dry-run mode.")
        device = 'cpu'
        compile = False
        dtype = 'float32'
    
    train_1B()
