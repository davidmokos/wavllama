import modal
import os
import torch
import torchaudio  # type: ignore
import sys
import datasets  # type: ignore
from tqdm import tqdm
import numpy as np

from model import GPTConfig, GPT


image = modal.Image.debian_slim(python_version="3.12") \
    .pip_install_from_requirements("requirements.txt") \


app = modal.App("gpt2-wavtokenizer", image=image)
vol = modal.Volume.from_name("wavllama-volume", create_if_missing=True)
mounts = [modal.Mount.from_local_dir("./WavTokenizer", remote_path="/root/WavTokenizer"),
          modal.Mount.from_local_dir("./examples", remote_path="/root/examples")]

def load_wavtokenizer_model():
    model_path = "/my_vol/wavtokenizer/wavtokenizer_medium_music_audio_320_24k_v2.ckpt"
    config_path = "/root/WavTokenizer/configs/wavtokenizer_mediumdata_music_audio_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sys.path.insert(0, "/root/WavTokenizer")
    from decoder.pretrained import WavTokenizer

    wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
    wavtokenizer = wavtokenizer.to(device)

    return wavtokenizer

@app.function(
    volumes={"/my_vol": vol},
    mounts=mounts,
    timeout=300,
    gpu="any",
    )
def infer():
    wavtokenizer = load_wavtokenizer_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load GPT-2 model and configuration
    ckpt_path = "/my_vol/gpt2_out/ckpt.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Initialize model
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    # Clean up state dict and load weights
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    # Set model to evaluation mode and move to device
    model.eval()
    model.to(device)
    
    # Generation parameters
    max_new_tokens = 500
    temperature = 0.8
    top_k = 200
    num_samples = 10
    
    # Generate tokens
    with torch.no_grad():
        # Start with empty context (or you could provide a starting sequence)
        x = torch.zeros((1, 1), dtype=torch.long, device=device)
        
        for k in range(num_samples):
            # Generate tokens
            generated_tokens = model.generate(
                x, 
                max_new_tokens, 
                temperature=temperature, 
                top_k=top_k
            )
            
            print(generated_tokens)
            
            print(generated_tokens.shape)
            
            print(f"Token range: {generated_tokens.min().item()} to {generated_tokens.max().item()}")
            generated_tokens = torch.clamp(generated_tokens, min=0, max=4095)
            print(f"Token range: {generated_tokens.min().item()} to {generated_tokens.max().item()}")
            
            
            # Convert to features using wavtokenizer
            features = wavtokenizer.codes_to_features(generated_tokens)
            
            print(features.shape)
            
            # Decode to audio
            audio_out = wavtokenizer.decode(
                features, 
                bandwidth_id=torch.tensor([0], device=device)
            )
            
            # Save the output audio
            output_file = f"/my_vol/generated_sample_{k}.wav"
            torchaudio.save(
                output_file,
                audio_out.cpu(),
                sample_rate=24000,
                encoding='PCM_S',
                bits_per_sample=16
            )
            print(f"Saved audio sample to {output_file}")
    
    
    
    


@app.local_entrypoint()
def main():
    infer.remote()