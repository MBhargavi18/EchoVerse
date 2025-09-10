# tts.py
import os
import re
import numpy as np
import soundfile as sf
from transformers import MarianMTModel, MarianTokenizer

# Load Telugu translation model
model_name = "Helsinki-NLP/opus-mt-en-te"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_to_telugu(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)


def clean_text_for_tts(text: str) -> str:
    """
    Remove unsupported characters before sending text to TTS.
    """
    return re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"-]", "", text)


def synthesize_speech(text: str, output_file: str) -> str:
    """
    Convert text into speech.
    Priority:
    1. Coqui TTS (best quality, requires `pip install TTS`)
    2. pyttsx3 (offline fallback)
    3. Simple sine wave audio (last resort)
    """
    safe_text = clean_text_for_tts(text)

    # --- Try Coqui TTS ---
    try:
        from TTS.api import TTS
        # Use a more robust model than Tacotron2
        model_name = "tts_models/en/ljspeech/glow-tts"

        tts = TTS(model_name, progress_bar=False)
        tts.tts_to_file(text=safe_text, file_path=output_file)
        return output_file
    except ImportError:
        print("⚠️ Coqui TTS not installed. Falling back to pyttsx3...")
    except Exception as e:
        print(f"⚠️ Coqui TTS failed: {e}. Falling back to pyttsx3...")

    # --- Try pyttsx3 ---
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 0.9)
        engine.save_to_file(safe_text, output_file)
        engine.runAndWait()
        return output_file
    except Exception as e:
        print(f"⚠️ pyttsx3 failed: {e}. Falling back to simple audio...")

    # --- Final Fallback ---
    return generate_simple_audio(safe_text, output_file)


def generate_simple_audio(text: str, output_file: str, sample_rate: int = 22050) -> str:
    """
    Generate a simple sine wave audio if no TTS engine is available.
    """
    words = len(text.split())
    duration = max(2.0, words * 0.4)  # ~0.4 sec per word

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    frequencies = [220, 330, 440, 550]
    audio = np.zeros_like(t)

    for i, freq in enumerate(frequencies):
        start_idx = int(i * len(t) / len(frequencies))
        end_idx = int((i + 1) * len(t) / len(frequencies))
        audio[start_idx:end_idx] += 0.3 * np.sin(2 * np.pi * freq * t[start_idx:end_idx])

    variation = 0.1 * np.sin(2 * np.pi * 2 * t)
    audio += variation

    audio = audio / np.max(np.abs(audio)) * 0.7
    fade_samples = int(0.1 * sample_rate)
    audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
    audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)

    sf.write(output_file, audio, sample_rate)
    return output_file


def check_tts_availability() -> bool:
    """
    Check if a usable TTS engine is available.
    """
    try:
        from TTS.api import TTS
        return True
    except ImportError:
        try:
            import pyttsx3
            return True
        except ImportError:
            return False


if __name__ == "__main__":
    test_text = "Hello, this is a test of the EchoVerse text-to-speech system."
    output_path = "test_output.wav"

    print(f"TTS Available: {check_tts_availability()}")
    try:
        result = synthesize_speech(test_text, output_path)
        print(f"✅ Audio generated: {result}")
        if os.path.exists(output_path):
            print(f"File size: {os.path.getsize(output_path)} bytes")
    except Exception as e:
        print(f"❌ Error: {e}")
