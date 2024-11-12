
import modal
import os
import torch
import torchaudio  # type: ignore
import sys
import datasets  # type: ignore
from tqdm import tqdm
import numpy as np


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


@app.function(volumes={"/my_vol": vol}, mounts=mounts, timeout=2*3600)
def test_wavtokenizer_model():

    file_name = "DXeiJpZXVAI"

    wavtokenizer = load_wavtokenizer_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess the audio file
    input_file = f"/root/examples/{file_name}.wav"
    wav, sr = torchaudio.load(input_file, format="wav")
    wav = torchaudio.functional.resample(wav, sr, 24000)  # Resample to 24kHz
    wav = wav.to(device)

    # Encode the audio to tokens
    with torch.no_grad():
        features, discrete_code = wavtokenizer.encode_infer(
            wav, bandwidth_id=torch.tensor([0], device=device))

    print(features.shape)
    # torch.Size([2, 512, 8588])
    print(discrete_code.shape)
    # torch.Size([1, 2, 8588])
    print(discrete_code)
    # tensor([[[4062, 1269,  753,  ..., 3655,  270, 2662],
    #          [4062, 1293, 1462,  ..., 1400, 3880, 2662]]])

    # Decode the tokens back to audio

    features2 = wavtokenizer.codes_to_features(discrete_code)
    print("features2.shape", features2.shape)
    # torch.Size([2, 512, 8588])

    features3 = features2[0]
    features3 = features3.unsqueeze(0)
    print("features3.shape", features3.shape)
    # torch.Size([1, 512, 8588])

    # features4 = features2[1]
    # features4 = features4.unsqueeze(0)
    # print("features4.shape",features4.shape)
    # torch.Size([1, 512, 8588])

    with torch.no_grad():
        audio_out = wavtokenizer.decode(
            features2, bandwidth_id=torch.tensor([0], device=device))

    # Save the output audio file
    output_file = f"/my_vol/{file_name}_out.wav"
    torchaudio.save(output_file, audio_out.cpu(),
                    sample_rate=24000, encoding='PCM_S', bits_per_sample=16)

    with torch.no_grad():
        audio_out = wavtokenizer.decode(
            features3, bandwidth_id=torch.tensor([0], device=device))

    # Save the output audio file
    output_file = f"/my_vol/{file_name}_out3.wav"
    torchaudio.save(output_file, audio_out.cpu(),
                    sample_rate=24000, encoding='PCM_S', bits_per_sample=16)

    # with torch.no_grad():
    #     audio_out = wavtokenizer.decode(
    #         features4, bandwidth_id=torch.tensor([0], device=device))

    # # Save the output audio file
    # output_file = f"/my_vol/{file_name}_out4.wav"
    # torchaudio.save(output_file, audio_out.cpu(),
    #                 sample_rate=24000, encoding='PCM_S', bits_per_sample=16)

    print(f"Processed audio saved to {output_file}")

    print(os.listdir("/my_vol"))


@app.function(volumes={"/my_vol": vol}, timeout=2*3600)
def download_dataset():

    dataset_dir = "/my_vol/amaai-lab-musicbench"

    # skip if already downloaded
    if os.path.exists(f"{dataset_dir}/MusicBench.tar.gz"):
        print("Dataset already downloaded")
    else:
        from huggingface_hub import snapshot_download # type: ignore
        snapshot_download("amaai-lab/MusicBench",
                          repo_type="dataset", local_dir=dataset_dir)
        print("Dataset downloaded")

    if os.path.exists(f"{dataset_dir}/datashare"):
        print("Dataset already extracted")
    else:
        os.system(f"tar -xzf {dataset_dir}/MusicBench.tar.gz -C {dataset_dir}")


def preprocess_function_wav_only(examples, dataset_dir):
    # wavtokenizer_tokens = []
    # processed_captions = []

    wavs = []

    files = [f"{dataset_dir}/{loc}" for loc in examples["location"]]
    for i, file in enumerate(files):
        wav, sr = torchaudio.load(file)
        wav = torchaudio.functional.resample(
            wav, sr, 24000)  # Resample to 24kHz
        # wav = wav.to(device)

        wavs.append(wav.numpy())

        # Encode the audio to tokens
        # with torch.no_grad():
        #     features, codes = wavtokenizer.encode_infer(wav, bandwidth_id=torch.tensor([0], device=device))

        # wavtokenizer_tokens.append(features.cpu().numpy())
        # processed_captions.append(examples['main_caption'][i])

    return {
        'wavs': wavs,
        # 'location': examples['location']
    }


@app.function(volumes={"/my_vol": vol}, mounts=mounts, timeout=24*3600, secrets=[modal.Secret.from_name("huggingface-secret-david")])
def preprocess_wav_only(num_examples=None):

    dataset_dir = "/my_vol/amaai-lab-musicbench/datashare"
    assert os.path.exists(dataset_dir)

    # wavtokenizer = load_wavtokenizer_model()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = datasets.load_dataset(
        "amaai-lab/MusicBench", cache_dir="/my_vol/hf_cache")
    print("Original dataset features:", dataset['train'].features)

    # Select only the first num_examples for each split
    if num_examples is not None:
        dataset = datasets.DatasetDict({
            split: dataset[split].select(
                range(min(num_examples, len(dataset[split]))))
            for split in dataset.keys()
        })

    def preprocess_fcn_call(examples):
        return preprocess_function_wav_only(examples, dataset_dir)

    # Process the dataset
    processed_dataset = dataset.map(
        preprocess_fcn_call,
        batched=True,
        batch_size=32,  # Adjust this based on your memory constraints
        # remove_columns=dataset['train'].column_names,
        desc="Processing audio files"
    )

    print("Processed dataset features:", processed_dataset['train'].features)

    processed_dataset.push_to_hub("davidmokos/musicbench-wav")
    
    
    
def preprocess_function_wavtokenizer(examples, wavtokenizer, device):
    wavtokenizer_tokens = []
    wavs = examples['wavs']

    for wav in wavs:
        with torch.no_grad():
            wav = torch.tensor(wav).to(device)
            _, codes = wavtokenizer.encode_infer(wav, bandwidth_id=torch.tensor([0], device=device))
            wavtokenizer_tokens.append(codes.cpu().numpy())

    return {
        'discrete_codes': wavtokenizer_tokens,
        # 'location': examples['location']
    }

@app.function(
    volumes={"/my_vol": vol},
    mounts=mounts,
    timeout=16*3600, 
    secrets=[modal.Secret.from_name("huggingface-secret-david")],
    gpu="any"
    )
def preprocess_wavtokenizer():
    dataset = datasets.load_dataset("davidmokos/musicbench-wav", cache_dir="/my_vol/hf_cache")
    print("Original dataset features:", dataset['train'].features)
    
    wavtokenizer = load_wavtokenizer_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def preprocess_fcn_call(examples):
        return preprocess_function_wavtokenizer(examples, wavtokenizer, device)

    # Process the dataset
    processed_dataset = dataset.map(
        preprocess_fcn_call,
        batched=True,
        batch_size=16,  # Adjust this based on your memory constraints
        # remove_columns=dataset['train'].column_names,
        desc="Wavtokenizer"
    )

    print("Processed dataset features:", processed_dataset['train'].features)

    processed_dataset.push_to_hub("davidmokos/musicbench-wavtokenizer")
    
@app.function(
    volumes={"/my_vol": vol},
    mounts=mounts,
    timeout=16*3600, 
    secrets=[modal.Secret.from_name("huggingface-secret-david")],
    )
def preprocess_get_gpt2_dataset():
    num_proc = 8
    dataset = datasets.load_dataset("davidmokos/musicbench-wavtokenizer", cache_dir="/my_vol/hf_cache", num_proc=num_proc)
    
    split_dataset = dataset["train"].train_test_split(test_size=0.01, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val
    
    print(split_dataset)
    
    def process(example):
        ids = example['discrete_codes']
        # it's either torch.Size([1, 2, audio_length]) or torch.Size([1, 1, audio_length])
        # something like 
        # tensor([[[4062, 1269,  753,  ..., 3655,  270, 2662],
        #          [4062, 1293, 1462,  ..., 1400, 3880, 2662]]])
        # we want to only take the first one
        ids = list(ids[0][0])
        
        ids.append(4096) # add the end of text token ???
        print(ids)
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=split_dataset['train'].column_names,
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
        
    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        dir_name = "/my_vol/musicbench-gpt2"
        os.makedirs(dir_name, exist_ok=True)
        filename = os.path.join(dir_name, f'{split}.bin')
        dtype = np.uint16 # (can do since vocab size is 4096)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
    
    

@app.local_entrypoint()
def main():
    # download_dataset.remote()
    # test_wavtokenizer_model.remote()
    # preprocess_wavtokenizer.remote()
    preprocess_get_gpt2_dataset.remote()
