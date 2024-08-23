# text_to_speech.py (Main File)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import soundfile as sf
import sounddevice as sd
from TTS.api import TTS
from speech_to_text import detect_and_record_audio, transcribe_audio
from query_generation import generate_response


# Initialize the TTS model once
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True)

def synthesize_text(text):
    print("Synthesizing speech with Coqui TTS...")

    # Generate audio from text
    wav_output = tts.tts(text)

    # Convert the output to a numpy array
    audio_array = np.array(wav_output)

    # Save the generated audio to a file
    sf.write("output.wav", audio_array, 16000)  # Sample rate is 22kHz for this model

    # Play the audio
    sd.play(audio_array, samplerate=16000)
    sd.wait()

def process_voice_query():
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

        # Step 4: Synthesize the generated response to speech
        synthesize_text(response_text)

if __name__ == "__main__":
    process_voice_query()
