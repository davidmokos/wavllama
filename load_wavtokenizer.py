import torch
import sys
import os
from huggingface_hub import snapshot_download  # type: ignore

# Add the WavTokenizer directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
wavtokenizer_dir = os.path.join(script_dir, 'WavTokenizer')
sys.path.insert(0, wavtokenizer_dir)

from decoder.pretrained import WavTokenizer

def load_wavtokenizer():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the WavTokenizer model
    config_path = "./WavTokenizer/configs/wavtokenizer_mediumdata_music_audio_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    model_id = "novateur/WavTokenizer-medium-music-audio-75token"

    # Download the entire model repository
    model_dir = snapshot_download(repo_id=model_id)

    # Find the model file (assuming it ends with .ckpt or .pt)
    model_files = [f for f in os.listdir(model_dir) if f.endswith(('.ckpt', '.pt'))]
    if not model_files:
        raise FileNotFoundError(f"No model file found in {model_dir}")
    model_path = os.path.join(model_dir, model_files[0])

    wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
    wavtokenizer = wavtokenizer.to(device)
    
    return wavtokenizer
