import modal
import os


image = modal.Image.debian_slim(python_version="3.12") \
    .pip_install_from_requirements("requirements.txt") \

app = modal.App("wavllama-download-models", image=image)
vol = modal.Volume.from_name("wavllama-volume")


@app.function(volumes={"/my_vol": vol}, timeout=2*3600)
def download_wavtokenizer_model():
    model_id = "novateur/WavTokenizer-medium-music-audio-75token"
    from huggingface_hub import snapshot_download  # type: ignore
    model_dir = snapshot_download(
        repo_id=model_id, local_dir="/my_vol/wavtokenizer")
    print(f"Model downloaded to {model_dir}")


@app.function(volumes={"/my_vol": vol}, secrets=[modal.Secret.from_name("huggingface-secret-david")], timeout=2*3600)
def download_llama_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token=os.environ["HF_TOKEN"], cache_dir="/my_vol/llama")
    print(tokenizer)
    
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", token=os.environ["HF_TOKEN"], cache_dir="/my_vol/llama")
    print(model)


@app.local_entrypoint()
def main():
    download_llama_model.remote()
    download_wavtokenizer_model.remote()
