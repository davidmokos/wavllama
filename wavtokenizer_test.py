import torch
import torchaudio  # type: ignore
import sys
import os

from load_wavtokenizer import load_wavtokenizer


wavtokenizer = load_wavtokenizer()

# Load and preprocess the audio file
input_file = "./examples/file_example_WAV_1MG.wav"
wav, sr = torchaudio.load(input_file)
wav = torchaudio.functional.resample(wav, sr, 24000)  # Resample to 24kHz
wav = wav.to(wavtokenizer.device)

# Encode the audio to tokens
with torch.no_grad():
    features, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=torch.tensor([0]))

# Decode the tokens back to audio
with torch.no_grad():
    audio_out = wavtokenizer.decode(features, bandwidth_id=torch.tensor([0]))

# Save the output audio file
output_file = "file_example_WAV_1MG_out.wav"
torchaudio.save(output_file, audio_out.cpu(), sample_rate=24000, encoding='PCM_S', bits_per_sample=16)

print(f"Processed audio saved to {output_file}")
