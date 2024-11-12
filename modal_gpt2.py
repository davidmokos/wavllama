import modal
import os
import torch
import torchaudio  # type: ignore
import sys
import datasets  # type: ignore

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT


image = modal.Image.debian_slim(python_version="3.12") \
    .pip_install_from_requirements("requirements.txt") \


app = modal.App("gpt2-wavtokenizer", image=image)
vol = modal.Volume.from_name("wavllama-volume", create_if_missing=True)
mounts = [modal.Mount.from_local_dir("./WavTokenizer", remote_path="/root/WavTokenizer"),
          modal.Mount.from_local_dir("./examples", remote_path="/root/examples")]







@app.function(
    volumes={"/my_vol": vol},
    mounts=mounts,
    timeout=24*3600, 
    secrets=[modal.Secret.from_name("huggingface-secret-david")],
    # gpu="any"
    )
def train_gpt2():
    print("Training GPT2")
    
    dataset = datasets.load_dataset("davidmokos/musicbench-wavtokenizer-small", cache_dir="/my_vol/hf_cache")
    
    
    
    print("Original dataset features:", dataset['train'].features)
    
    print(type(dataset['train'].data['discrete_codes']))
    
    print(type(dataset['train'].data['discrete_codes'][0]))
    
    
    


@app.local_entrypoint()
def main():
    # download_dataset.remote()
    # test_wavtokenizer_model.remote()
    train_gpt2.remote()