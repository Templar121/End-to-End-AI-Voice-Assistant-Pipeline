import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import soundfile as sf
import sounddevice as sd
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from speech_to_text import detect_and_record_audio, transcribe_audio
from query_generation import generate_response
from datasets import load_dataset
from torchaudio.transforms import Resample
from transformers import SpeechT5HifiGan  
import torchaudio

# Initialize the SpeechT5 model and processor
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan") 

# Load dataset for speaker embeddings
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

# Map speaker embeddings to gender/voice types
voice_options = {
    "Joanna": torch.tensor(embeddings_dataset[0]["xvector"]).unsqueeze(0),  # female voice
    "Samantha": torch.tensor(embeddings_dataset[1]["xvector"]).unsqueeze(0),  # female voice
    "Matthew": torch.tensor(embeddings_dataset[2]["xvector"]).unsqueeze(0),  # male voice
    "Brian": torch.tensor(embeddings_dataset[3]["xvector"]).unsqueeze(0),  # male voice
}

def synthesize_text(text, voice="Joanna", speed=1.0, pitch=1.0):
    print(f"Synthesizing speech with {voice} using SpeechT5...")

    # Step 1: Process the input text
    inputs = processor(text=text, return_tensors="pt")

    # Step 2: Get the speaker embedding based on selected voice
    speaker_embeddings = voice_options.get(voice, voice_options["Joanna"])

    # Step 3: Generate spectrogram using the TTS model
    with torch.no_grad():
        spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)
        speech = vocoder(spectrogram).squeeze()

    # Step 4: Convert the speech to a numpy array
    audio_array = speech.cpu().numpy()

    # Step 5: Resample if needed (e.g., to match your playback system)
    resampler = Resample(orig_freq=24000, new_freq=16000)
    audio_array = resampler(torch.tensor(audio_array)).numpy()

    # Step 6: Apply VAD to remove silent sections
    audio_tensor = torch.tensor(audio_array)
    vad = torchaudio.transforms.Vad(sample_rate=16000)
    cleaned_audio = vad(audio_tensor).numpy()

    # Step 7: Save the generated and cleaned audio to a file
    sf.write("output_vad.wav", cleaned_audio, 16000)

    # Step 8: Play the cleaned audio
    sd.play(cleaned_audio, samplerate=16000)
    sd.wait()

def process_voice_query(voice="Joanna", speed=1.0, pitch=1.0):
    print("Starting the process...")

    # Step 1: Record and detect speech
    audio_data = detect_and_record_audio()

    if audio_data is not None:
        # Step 2: Transcribe the detected speech to text
        query_text = transcribe_audio(audio_data)
        print(f"User Query: {query_text}")

        # Step 3: Generate response based on the transcribed query
        response_text = generate_response(query_text)
        print(f"Response: {response_text}")

        # Step 4: Synthesize the generated response to speech with the selected voice, speed, and pitch
        synthesize_text(response_text, voice=voice, speed=speed, pitch=pitch)

if __name__ == "__main__":
    # Choose voice: "Joanna", "Samantha", "Matthew", "Brian"
    process_voice_query(voice="Samantha", speed=1.0, pitch=1.0)
