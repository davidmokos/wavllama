import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import BertModel, BertTokenizer # type: ignore
from datasets import load_dataset # type: ignore
import wandb
from einops import rearrange, repeat
import math
from tqdm import tqdm

# Define Modal setup
vol = modal.Volume.from_name("wavllama-volume", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements("requirements.txt")
app = modal.App("wavllama-training", image=image)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, heads=4, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        
    def forward(self, x, context):
        h = self.heads
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FlowBlock(nn.Module):
    def __init__(self, dim, context_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        self.self_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.cross_attn = CrossAttention(dim, context_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x, context):
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.mlp(self.norm3(x))
        return x

class MusicFlowMatching(nn.Module):
    def __init__(self, hidden_dim=256, sequence_length=150, num_layers=4):
        super().__init__()
        
        # Text encoder (frozen BERT)
        self.bert = BertModel.from_pretrained('prajjwal1/bert-tiny', cache_dir="/my_vol/bert-tiny")
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Project BERT features to hidden dim
        self.text_projection = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        
        # Time embedding
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        
        # Position embedding for token sequence
        self.pos_embedding = nn.Parameter(torch.randn(1, sequence_length, hidden_dim))
        
        # Initial projection for noise
        self.noise_projection = nn.Linear(1, hidden_dim)
        
        # Flow layers
        self.layers = nn.ModuleList([
            FlowBlock(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Final velocity prediction
        self.to_velocity = nn.Linear(hidden_dim, 1)
        
    def forward(self, noise, time, text_ids, attention_mask):
        # Get text embeddings from BERT
        with torch.no_grad():
            text_features = self.bert(text_ids, attention_mask=attention_mask)[0]
        text_features = self.text_projection(text_features)
        
        # Time embedding
        t_emb = self.time_embedding(time)
        t_emb = repeat(t_emb, 'b d -> b n d', n=noise.shape[1])
        
        # Project noise and add position embedding
        x = self.noise_projection(noise.unsqueeze(-1))
        x = x + self.pos_embedding + t_emb
        
        # Flow layers
        for layer in self.layers:
            x = layer(x, text_features)
        
        # Predict velocity
        velocity = self.to_velocity(x).squeeze(-1)
        return velocity

    def get_velocity(self, x0, x1, t):
        """Calculate target velocity given two points and time"""
        return (x1 - x0) / t

class MusicDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_text_length=512, sequence_length=150):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.dataset)
    
    def pad_or_truncate_tokens(self, tokens):
        # Convert to tensor if it's not already
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens)
        
        # Remove batch dimension if it exists
        if tokens.dim() == 3:
            tokens = tokens.squeeze(0)  # Remove batch dim
        if tokens.dim() == 2:
            tokens = tokens[0]  # Take first row if 2D
            
        current_length = tokens.shape[0]
        
        if current_length > self.sequence_length:
            return tokens[:self.sequence_length]
        elif current_length < self.sequence_length:
            padding = torch.zeros(self.sequence_length - current_length, dtype=tokens.dtype)
            return torch.cat([tokens, padding])
        return tokens
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        text_tokens = self.tokenizer(
            item['main_caption'],
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )
        
        wav_tokens = self.pad_or_truncate_tokens(item['wavtokenizer_tokens'])
        
        return {
            'input_ids': text_tokens['input_ids'].squeeze(0),
            'attention_mask': text_tokens['attention_mask'].squeeze(0),
            'wav_tokens': wav_tokens
        }

class MusicFlowMatchingLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        self.model = MusicFlowMatching(
            hidden_dim=config['hidden_dim'],
            sequence_length=config['sequence_length'],
            num_layers=config['num_layers']
        )
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target_tokens = batch['wav_tokens']
        
        t = torch.rand(input_ids.shape[0], device=self.device)
        noise = torch.randn_like(target_tokens, device=self.device)
        
        t_expanded = t[:, None]
        x_t = t_expanded * target_tokens + (1 - t_expanded) * noise
        
        target_velocity = self.model.get_velocity(noise, target_tokens, t_expanded)
        predicted_velocity = self.model(x_t, t, input_ids, attention_mask)
        
        loss = F.mse_loss(predicted_velocity, target_velocity)
        
        # Log gradient norm
        if batch_idx % 10 == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.trainer.gradient_clip_val)
            self.log('grad_norm', grad_norm)
        
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target_tokens = batch['wav_tokens']
        
        t = torch.rand(input_ids.shape[0], device=self.device)
        noise = torch.randn_like(target_tokens, device=self.device)
        
        t_expanded = t[:, None]
        x_t = t_expanded * target_tokens + (1 - t_expanded) * noise
        
        target_velocity = self.model.get_velocity(noise, target_tokens, t_expanded)
        predicted_velocity = self.model(x_t, t, input_ids, attention_mask)
        
        loss = F.mse_loss(predicted_velocity, target_velocity)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=3,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

@app.function(
    volumes={"/my_vol": vol},
    gpu="h100",
    timeout=24*3600,
    secrets=[modal.Secret.from_name("david-wandb-secret"), modal.Secret.from_name("huggingface-secret-david")],
    # retries=modal.Retries(max_retries=10, initial_delay=0.0)
)
def train():
    config = {
        "model_name": "music-flow-tiny",
        "hidden_dim": 256,
        "num_layers": 4,
        "batch_size": 32,
        "learning_rate": 1e-5,
        "num_epochs": 100,
        "sequence_length": 150,
        "max_text_length": 512
    }
    
    wandb_logger = WandbLogger(project="music-flow-matching", config=config)
    
    dataset = load_dataset(
        "davidmokos/musicbench-processed",
        cache_dir="/my_vol/musicbench"
    )
    dataset = dataset['train'].select(range(1000))
    dataset = dataset.train_test_split(test_size=0.1)
    
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny', cache_dir="/my_vol/bert-tiny")
    
    train_dataset = MusicDataset(
        dataset['train'],
        tokenizer,
        max_text_length=config['max_text_length'],
        sequence_length=config['sequence_length']
    )
    val_dataset = MusicDataset(
        dataset['test'],
        tokenizer,
        max_text_length=config['max_text_length'],
        sequence_length=config['sequence_length']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )
    
    model = MusicFlowMatchingLightning(config)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="/my_vol/checkpoints",
        filename="music-flow-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    
    trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],
        default_root_dir="/my_vol/lightning_logs",
        # gradient_clip_val=1.0,
        gradient_clip_val=0.5,  # Lower clip value
        gradient_clip_algorithm="norm",
        # precision="16-mixed",  # Add mixed precision training (crashes)
    )
    
    try:
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e
    finally:
        wandb.finish()

@app.local_entrypoint()
def main():
    train.remote()