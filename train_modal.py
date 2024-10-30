import modal
import os
import torch
import torchaudio  # type: ignore
import sys


image = modal.Image.debian_slim(python_version="3.12") \
    .pip_install_from_requirements("requirements.txt") \


app = modal.App("example-long-training-lightning", image=image)
vol = modal.Volume.from_name("wavllama-volume")


@app.function(
    volumes={"/my_vol": vol},
    mounts=[modal.Mount.from_local_dir("./WavTokenizer", remote_path="/root/WavTokenizer"),
            modal.Mount.from_local_dir("./examples", remote_path="/root/examples")]
)
def some_func():

    wavtokenizer = load_wavtokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess the audio file
    input_file = "/root/examples/file_example_WAV_1MG.wav"
    wav, sr = torchaudio.load(input_file, format="wav")
    wav = torchaudio.functional.resample(wav, sr, 24000)  # Resample to 24kHz
    wav = wav.to(device)

    # Encode the audio to tokens
    with torch.no_grad():
        features, discrete_code = wavtokenizer.encode_infer(
            wav, bandwidth_id=torch.tensor([0], device=device))

    # Decode the tokens back to audio
    with torch.no_grad():
        audio_out = wavtokenizer.decode(
            features, bandwidth_id=torch.tensor([0], device=device))

    # Save the output audio file
    output_file = "/my_vol/file_example_WAV_1MG_out.wav"
    torchaudio.save(output_file, audio_out.cpu(),
                    sample_rate=24000, encoding='PCM_S', bits_per_sample=16)

    print(f"Processed audio saved to {output_file}")

    print(os.listdir("/my_vol"))


@app.function(volumes={"/my_vol": vol}, timeout=2*3600)
def download_wavtokenizer_model():
    model_id = "novateur/WavTokenizer-medium-music-audio-75token"
    from huggingface_hub import snapshot_download  # type: ignore
    model_dir = snapshot_download(
        repo_id=model_id, cache_dir="/my_vol/wavtokenizer")
    print(f"Model downloaded to {model_dir}")


def load_wavtokenizer():
    model_path = "/my_vol/wavtokenizer/wavtokenizer_medium_music_audio_320_24k_v2.ckpt"
    config_path = "/root/WavTokenizer/configs/wavtokenizer_mediumdata_music_audio_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sys.path.insert(0, "/root/WavTokenizer")
    from decoder.pretrained import WavTokenizer
    
    wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
    wavtokenizer = wavtokenizer.to(device)

    return wavtokenizer


@app.function(volumes={"/my_vol": vol}, mounts=[modal.Mount.from_local_dir("./WavTokenizer", remote_path="/root/WavTokenizer")])
def check_wavtokenizer_model():
    load_wavtokenizer()
    
    
    
@app.function(
    volumes={"/my_vol": vol},
    mounts=[modal.Mount.from_local_dir("./WavTokenizer", remote_path="/root/WavTokenizer"),
            modal.Mount.from_local_dir("./examples", remote_path="/root/examples")]
)
def train_wavllama():
    pass


@app.function(volumes={"/my_vol": vol}, secrets=[modal.Secret.from_name("huggingface-secret-david")], timeout=2*3600)
def download_llama_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore
    # from transformers.utils import move_cache

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token=os.environ["HF_TOKEN"], cache_dir="/my_vol/llama")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", token=os.environ["HF_TOKEN"], cache_dir="/my_vol/llama")
    
    print(tokenizer)
    print(model)




@app.local_entrypoint()
def main():
    download_llama_model.remote()
    # download_wavtokenizer_model.remote()
