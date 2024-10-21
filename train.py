import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
import os
from tqdm import tqdm # type: ignore

class TextAudioDataset(Dataset):
    def __init__(self, metadata_file, tokens_dir, tokenizer, max_length=512):
        self.data = []
        self.tokens_dir = tokens_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(metadata_file, 'r') as f:
            for line in f:
                audio_path, caption, idx = line.strip().split('\t')
                self.data.append((audio_path, caption, int(idx)))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        _, caption, token_idx = self.data[idx]
        
        # Tokenize text
        text_tokens = self.tokenizer(caption, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        
        # Load audio tokens
        audio_tokens = torch.load(os.path.join(self.tokens_dir, f"audio_tokens_{token_idx}.pt"))
        
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
        for name, param in self.llama_model.named_parameters():
            if "layers.-1." not in name:  # Not the last layer
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # Add new layers
        self.projection = nn.Linear(self.llama_model.config.hidden_size, wav_tokenizer_vocab_size)
        self.layer_norm = nn.LayerNorm(wav_tokenizer_vocab_size)
        
    def forward(self, input_ids, attention_mask):
        # Get the output from Llama model
        outputs = self.llama_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Use the last hidden state
        last_hidden_state = outputs.hidden_states[-1]
        
        # Project to WavTokenizer vocabulary size
        projected = self.projection(last_hidden_state)
        
        # Apply layer normalization
        normalized = self.layer_norm(projected)
        
        return normalized

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        audio_tokens = batch['audio_tokens'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.view(-1, outputs.size(-1)), audio_tokens.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set hyperparameters
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10
    MAX_LENGTH = 512

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    # Load dataset
    dataset = TextAudioDataset("preprocessed_musicbench/metadata.txt", "preprocessed_musicbench", tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model
    llama_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    wav_tokenizer_vocab_size = 4096  # Assume this is the vocab size, adjust as needed
    model = TextToAudioModel(llama_model_name, wav_tokenizer_vocab_size).to(device)

    # Set up optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "text_to_audio_model.pth")
    print("Training completed. Model saved.")

if __name__ == "__main__":
    main()