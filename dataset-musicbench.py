from datasets import load_dataset, DatasetDict # type: ignore
import sys
import os
import torch

import torchaudio # type: ignore

from load_wavtokenizer import load_wavtokenizer
wavtokenizer = load_wavtokenizer()

dataset_dir = "/tmp/dataset-musicbench-files"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



def download_musicbench():
    # seems like we need to manually download the dataset as the original authors have uploaded it zipped
    url = "https://huggingface.co/datasets/amaai-lab/MusicBench/resolve/main/MusicBench.tar.gz?download=true"
    
    os.makedirs(dataset_dir, exist_ok=True)
    
    # skip if already downloaded
    if os.path.exists(f"{dataset_dir}/MusicBench.tar.gz"):
        print("Dataset already downloaded")
    else:
        os.system(f"curl -L {url} -o {dataset_dir}/MusicBench.tar.gz")
    
    if os.path.exists(f"{dataset_dir}/datashare"):
        print("Dataset already extracted")
    else:
        os.system(f"tar -xzf {dataset_dir}/MusicBench.tar.gz -C {dataset_dir}")

def preprocess_function(examples, wavtokenizer=wavtokenizer):
    wavtokenizer_tokens = []
    processed_captions = []
    
    files = [f"{dataset_dir}/datashare/{loc}" for loc in examples["location"]]
    for i, file in enumerate(files):
        wav, sr = torchaudio.load(file)
        wav = torchaudio.functional.resample(wav, sr, 24000)  # Resample to 24kHz
        wav = wav.to(device)
        
        # Encode the audio to tokens
        with torch.no_grad():
            features, _ = wavtokenizer.encode_infer(wav, bandwidth_id=torch.tensor([0], device=device))
        
        wavtokenizer_tokens.append(features.cpu().numpy())
        processed_captions.append(examples['main_caption'][i])
    
    return {
        'wavtokenizer_tokens': wavtokenizer_tokens,
        'main_caption': processed_captions,
        'location': examples['location']
    }

def load_musicbench(num_examples=None):
    dataset = load_dataset("amaai-lab/MusicBench")
    print("Original dataset features:", dataset['train'].features)
    
    # Select only the first num_examples for each split
    if num_examples is not None:
        dataset = DatasetDict({
            split: dataset[split].select(range(min(num_examples, len(dataset[split]))))
            for split in dataset.keys()
        })
    
    # Process the dataset
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=8,  # Adjust this based on your memory constraints
        remove_columns=dataset['train'].column_names,
        desc="Processing audio files"
    )
    
    print("Processed dataset features:", processed_dataset['train'].features)
    return processed_dataset

if __name__ == "__main__":
    download_musicbench()
    # processed_dataset = load_musicbench(1)  # Process only the first 100 examples of each split
    # for split in processed_dataset.keys():
    #     print(f"Number of processed examples in {split}:", len(processed_dataset[split]))
    # print("Sample processed example:", processed_dataset['train'][0].keys())
    
    processed_dataset = load_musicbench()
    processed_dataset.push_to_hub("davidmokos/musicbench-processed")
