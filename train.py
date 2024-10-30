import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import wandb
import numpy as np

class TextAudioDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Tokenize text
        text_tokens = self.tokenizer(
            item['main_caption'], 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        # Convert wavtokenizer tokens to tensor
        audio_tokens = torch.tensor(item['wavtokenizer_tokens'])
        
        return {
            'input_ids': text_tokens['input_ids'].squeeze(0),
            'attention_mask': text_tokens['attention_mask'].squeeze(0),
            'audio_tokens': audio_tokens.squeeze(0)
        }

class TextToAudioModel(nn.Module):
    def __init__(self, llama_model_name, wav_tokenizer_vocab_size):
        super().__init__()
        
        # Load the Llama 3 model
        self.llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name)
        
        # Freeze all layers except the last one
        total_params = 0
        trainable_params = 0
        for name, param in self.llama_model.named_parameters():
            total_params += param.numel()
            if "layers.-1." not in name:  # Not the last layer
                param.requires_grad = False
            else:
                param.requires_grad = True
                trainable_params += param.numel()
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
        
        # Add new layers
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
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        current_loss = loss.item()
        
        # Log metrics
        wandb.log({
            "batch_loss": current_loss,
            "epoch": epoch,
            "batch": batch_idx,
            "global_step": epoch * len(dataloader) + batch_idx
        })
        
        # Update progress bar
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

def main():
    # Initialize wandb
    wandb.init(
        project="text-to-audio-llama",
        config={
            "architecture": "Llama-3-8B + WavTokenizer",
            "dataset": "musicbench-processed",
            "learning_rate": 1e-4,
            "batch_size": 8,
            "epochs": 10,
            "max_length": 512
        }
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = load_dataset("davidmokos/musicbench-processed")
    
    # Create train/val split
    dataset = dataset['train'].train_test_split(test_size=0.1)
    train_dataset = dataset['train']
    val_dataset = dataset['test']

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    # Create dataloaders
    train_dataset = TextAudioDataset(train_dataset, tokenizer, wandb.config.max_length)
    val_dataset = TextAudioDataset(val_dataset, tokenizer, wandb.config.max_length)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=wandb.config.batch_size, 
        shuffle=True,
        num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=wandb.config.batch_size, 
        shuffle=False,
        num_workers=4
    )

    # Initialize model
    model = TextToAudioModel(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        wav_tokenizer_vocab_size=4096  # Adjust if needed
    ).to(device)
    
    # Log model architecture
    wandb.watch(model, log_freq=100)

    # Set up optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=wandb.config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(wandb.config.epochs):
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device, epoch)
        
        # Validate
        val_loss = validate(model, val_dataloader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, "best_model.pth")
            wandb.save("best_model.pth")
        
        print(f"Epoch {epoch+1}/{wandb.config.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
    
    wandb.finish()

if __name__ == "__main__":
    main()