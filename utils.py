# utils.py
import streamlit as st
import os
import logging
# utils.py (add these imports at the top)
import docx
import PyPDF2

def read_file(uploaded_file):
    """
    Read content from uploaded file (txt, pdf, docx)
    """
    try:
        if uploaded_file.name.endswith(".txt"):
            return uploaded_file.getvalue().decode("utf-8")

        elif uploaded_file.name.endswith(".docx"):
            doc = docx.Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])

        elif uploaded_file.name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text

        else:
            raise ValueError("Unsupported file format. Please upload .txt, .docx, or .pdf")

    except Exception as e:
        raise ValueError(f"Could not read file: {e}")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_txt(uploaded_file):
    """
    Read content from uploaded text file with proper error handling
    """
    try:
        # Read the file content
        content = uploaded_file.getvalue().decode("utf-8")
        
        # Basic validation
        if not content.strip():
            raise ValueError("File appears to be empty")
        
        # Log successful read
        logger.info(f"Successfully read file: {uploaded_file.name}, size: {len(content)} characters")
        
        return content
        
    except UnicodeDecodeError as e:
        logger.error(f"Unicode decode error: {e}")
        raise ValueError("File encoding not supported. Please use UTF-8 encoded text files.")
    
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise ValueError(f"Could not read file: {str(e)}")

def validate_text_input(text: str, min_length: int = 10, max_length: int = 10000) -> bool:
    """
    Validate text input for processing
    """
    if not text or not text.strip():
        return False, "Text cannot be empty"
    
    if len(text.strip()) < min_length:
        return False, f"Text must be at least {min_length} characters long"
    
    if len(text) > max_length:
        return False, f"Text cannot exceed {max_length} characters"
    
    return True, "Valid"

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

def get_text_statistics(text: str) -> dict:
    """
    Get basic statistics about the text
    """
    words = text.split()
    characters = len(text)
    characters_no_spaces = len(text.replace(" ", ""))
    sentences = text.count(".") + text.count("!") + text.count("?")
    paragraphs = len([p for p in text.split("\n\n") if p.strip()])
    
    # Estimate reading time (average 200 words per minute)
    estimated_reading_time = len(words) / 200
    
    # Estimate audio duration (average 150 words per minute for speech)
    estimated_audio_duration = len(words) / 150
    
    return {
        "words": len(words),
        "characters": characters,
        "characters_no_spaces": characters_no_spaces,
        "sentences": max(1, sentences),  # At least 1
        "paragraphs": max(1, paragraphs),  # At least 1
        "estimated_reading_time_minutes": estimated_reading_time,
        "estimated_audio_duration_minutes": estimated_audio_duration
    }

def check_environment() -> dict:
    """
    Check environment setup and requirements
    """
    checks = {
        "streamlit": True,  # If we're running, streamlit is available
        "torch": False,
        "transformers": False,
        "tts": False,
        "hf_token": False,
        "soundfile": False
    }
    
    # Check PyTorch
    try:
        import torch
        checks["torch"] = True
    except ImportError:
        pass
    
    # Check Transformers
    try:
        import transformers
        checks["transformers"] = True
    except ImportError:
        pass
    
    # Check TTS
    try:
        from TTS.api import TTS
        checks["tts"] = True
    except ImportError:
        pass
    
    # Check HuggingFace token
    checks["hf_token"] = bool(os.getenv("HF_TOKEN"))
    
    # Check soundfile
    try:
        import soundfile
        checks["soundfile"] = True
    except ImportError:
        pass
    
    return checks

def display_environment_status():
    """
    Display environment setup status in Streamlit
    """
    checks = check_environment()
    
    st.subheader("🔧 Environment Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Core Dependencies:**")
        st.write(f"✅ Streamlit: {'Available' if checks['streamlit'] else '❌ Missing'}")
        st.write(f"{'✅' if checks['torch'] else '❌'} PyTorch: {'Available' if checks['torch'] else 'Missing'}")
        st.write(f"{'✅' if checks['transformers'] else '❌'} Transformers: {'Available' if checks['transformers'] else 'Missing'}")
        st.write(f"{'✅' if checks['soundfile'] else '❌'} SoundFile: {'Available' if checks['soundfile'] else 'Missing'}")
    
    with col2:
        st.write("**Optional Dependencies:**")
        st.write(f"{'✅' if checks['tts'] else '❌'} TTS Library: {'Available' if checks['tts'] else 'Missing'}")
        st.write(f"{'✅' if checks['hf_token'] else '❌'} HF Token: {'Set' if checks['hf_token'] else 'Not Set'}")
    
    # Show warnings for missing dependencies
    missing_core = [k for k, v in checks.items() if not v and k in ['torch', 'transformers', 'soundfile']]
    if missing_core:
        st.warning(f"⚠️ Missing core dependencies: {', '.join(missing_core)}")
        st.info("Install with: `pip install torch transformers soundfile`")
    
    if not checks['hf_token']:
        st.info("💡 Set HF_TOKEN environment variable for Granite model access")
    
    if not checks['tts']:
        st.info("💡 Install TTS library for better audio quality: `pip install TTS`")

# Test function
if __name__ == "__main__":
    print("Testing utils module...")
    
    # Test text statistics
    sample_text = "This is a sample text. It has multiple sentences! Does it work correctly?"
    stats = get_text_statistics(sample_text)
    print("Text statistics:", stats)
    
    # Test environment check
    env_status = check_environment()
    print("Environment status:", env_status)