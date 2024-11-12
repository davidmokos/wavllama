import modal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm
import os
import sys

# Define the Modal volume and app
vol = modal.Volume.from_name("wavllama-volume", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12") \
    .pip_install_from_requirements("requirements.txt") \

app = modal.App("wavllama-training", image=image)

class TextAudioDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        text_tokens = self.tokenizer(
            item['main_caption'], 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        audio_tokens = torch.tensor(item['wavtokenizer_tokens'])
        
        return {
            'input_ids': text_tokens['input_ids'].squeeze(0),
            'attention_mask': text_tokens['attention_mask'].squeeze(0),
            'audio_tokens': audio_tokens.squeeze(0)
        }

class TextToAudioModel(nn.Module):
    def __init__(self, llama_model, wav_tokenizer_vocab_size, num_trainable_layers=4):
        super().__init__()
        
        self.llama_model = llama_model
        
        # Count total parameters and freeze all initially
        total_params = 0
        for param in self.llama_model.parameters():
            total_params += param.numel()
            param.requires_grad = False
        
        # Unfreeze the last n transformer layers
        trainable_params = 0
        for i in range(num_trainable_layers):
            layer_idx = len(self.llama_model.model.layers) - 1 - i
            for param in self.llama_model.model.layers[layer_idx].parameters():
                param.requires_grad = True
                trainable_params += param.numel()
        
        # Also unfreeze the output projection layers
        for param in self.llama_model.lm_head.parameters():
            param.requires_grad = True
            trainable_params += param.numel()
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
        
        # Add new layers for audio token prediction
        self.projection = nn.Linear(self.llama_model.config.hidden_size, wav_tokenizer_vocab_size)
        self.layer_norm = nn.LayerNorm(wav_tokenizer_vocab_size)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.llama_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        projected = self.projection(last_hidden_state)
        normalized = self.layer_norm(projected)
        return normalized

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        audio_tokens = batch['audio_tokens'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.view(-1, outputs.size(-1)), audio_tokens.view(-1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        current_loss = loss.item()
        
        wandb.log({
            "batch_loss": current_loss,
            "epoch": epoch,
            "batch": batch_idx,
            "global_step": epoch * len(dataloader) + batch_idx
        })
        
        progress_bar.set_postfix({
            'loss': f"{current_loss:.4f}",
            'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
        })
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_tokens = batch['audio_tokens'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), audio_tokens.view(-1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

@app.function(
    volumes={"/my_vol": vol},
    mounts=[modal.Mount.from_local_dir("./WavTokenizer", remote_path="/root/WavTokenizer")],
    gpu="h100",
    timeout=24*3600,  # 24 hours
    secrets=[modal.Secret.from_name("david-wandb-secret"), modal.Secret.from_name("huggingface-secret-david")],
    # retries=modal.Retries(max_retries=10, initial_delay=0.0)
)
def train_wavllama():
    # Initialize wandb
    wandb.init(
        project="wavllama",
        config={
            "architecture": "Llama-3-8B + WavTokenizer",
            "dataset": "musicbench-processed",
            "learning_rate": 1e-4,
            "batch_size": 8,
            "epochs": 10,
            "max_length": 512,
            "trainable_layers": 4
        }
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset from cache
    from datasets import load_dataset # type: ignore
    dataset = load_dataset("davidmokos/musicbench-processed", cache_dir="/my_vol/musicbench")
    dataset = dataset['train'].train_test_split(test_size=0.1)
    
    # Load models from cache
    from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir="/my_vol/llama")
    tokenizer.pad_token = tokenizer.eos_token  # Use the end-of-sequence token as padding
    tokenizer.padding_side = "right"  # Pad on the right side of the sequence
    llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir="/my_vol/llama")

    # Load WavTokenizer
    sys.path.insert(0, "/root/WavTokenizer")
    from decoder.pretrained import WavTokenizer
    model_path = "/my_vol/wavtokenizer/wavtokenizer_medium_music_audio_320_24k_v2.ckpt"
    config_path = "/root/WavTokenizer/configs/wavtokenizer_mediumdata_music_audio_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
    wav_tokenizer_vocab_size = 4096  # Adjust if needed

    # Create dataloaders
    train_dataset = TextAudioDataset(dataset['train'], tokenizer, wandb.config.max_length)
    val_dataset = TextAudioDataset(dataset['test'], tokenizer, wandb.config.max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = TextToAudioModel(
        llama_model,
        wav_tokenizer_vocab_size=wav_tokenizer_vocab_size,
        num_trainable_layers=wandb.config.trainable_layers
    ).to(device)
    
    wandb.watch(model, log_freq=100)

    optimizer = optim.AdamW(model.parameters(), lr=wandb.config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    criterion = nn.CrossEntropyLoss()

    # Training loop with checkpointing
    best_val_loss = float('inf')
    checkpoint_path = "/my_vol/checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)
    
    for epoch in range(wandb.config.epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device, epoch)
        val_loss = validate(model, val_dataloader, criterion, device)
        scheduler.step(val_loss)
        
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        
        checkpoint_file = os.path.join(checkpoint_path, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_file)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_path, "best_model.pt")
            torch.save(checkpoint, best_model_path)
            wandb.save(best_model_path)
        
        print(f"Epoch {epoch+1}/{wandb.config.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
    
    wandb.finish()

@app.local_entrypoint()
def main():
    train_wavllama.remote()