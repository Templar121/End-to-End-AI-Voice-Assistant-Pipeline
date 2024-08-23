# speech_to_text.py
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import torch

# Initialize the Faster Whisper model
model = WhisperModel("small", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float32")

def detect_and_record_audio(duration=5, samplerate=16000):
    print("Listening for voice...")

    # Record audio for the given duration
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()

    # Return the flattened audio for processing
    return audio.flatten()

def transcribe_audio(audio):
    print("Transcribing...")

    # Convert audio to float32 and normalize
    audio = audio.astype(np.float32) / 32768.0  # Normalize 16-bit PCM audio to range [-1.0, 1.0]

    # Transcribe the audio using Faster Whisper model
    segments, _ = model.transcribe(audio, language="en")

    # Gather the transcription from all segments
    transcript = "".join([segment.text for segment in segments])
    return transcript
