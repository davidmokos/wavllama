import torch
from datasets import load_dataset
import sys
import os
import torchaudio
from tqdm import tqdm
from huggingface_hub import snapshot_download

# Add the WavTokenizer directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
wavtokenizer_dir = os.path.join(script_dir, 'WavTokenizer')
sys.path.insert(0, wavtokenizer_dir)

from decoder.pretrained import WavTokenizer

def preprocess_dataset(wav_tokenizer, dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav_tokenizer = wav_tokenizer.to(device)
    
    for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
        # Load audio
        wav, sr = torchaudio.load(item['location'])
        wav = torchaudio.functional.resample(wav, sr, 24000)  # Resample to 24kHz if necessary
        wav = wav.to(device)
        
        # Generate tokens
        bandwidth_id = torch.tensor([0]).to(device)
        with torch.no_grad():
            features, audio_tokens = wav_tokenizer.encode_infer(wav.unsqueeze(0), bandwidth_id=bandwidth_id)
        
        # Save tokens
        torch.save(audio_tokens, os.path.join(output_dir, f"audio_tokens_{idx}.pt"))
        
        # Save metadata (file path and main caption)
        with open(os.path.join(output_dir, "metadata.txt"), "a") as f:
            f.write(f"{item['location']}\t{item['main_caption']}\t{idx}\n")

def main():
    # Load dataset
    dataset = load_dataset("amaai-lab/MusicBench")
    
    # Set up WavTokenizer
    config_path = "./WavTokenizer/configs/wavtokenizer_mediumdata_music_audio_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    model_id = "novateur/WavTokenizer-medium-music-audio-75token"

    # Download the entire model repository
    model_dir = snapshot_download(repo_id=model_id)

    # Find the model file (assuming it ends with .ckpt or .pt)
    model_files = [f for f in os.listdir(model_dir) if f.endswith(('.ckpt', '.pt'))]
    if not model_files:
        raise FileNotFoundError(f"No model file found in {model_dir}")
    model_path = os.path.join(model_dir, model_files[0])

    # Load WavTokenizer
    wav_tokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
    
    # Preprocess dataset
    preprocess_dataset(wav_tokenizer, dataset['train'], "preprocessed_musicbench")

if __name__ == "__main__":
    main()
