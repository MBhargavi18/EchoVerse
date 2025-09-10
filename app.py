import streamlit as st
from pathlib import Path
from transformers import pipeline
from docx import Document
import pdfplumber
from TTS.api import TTS


st.title("📚 EchoVerse – AI Audiobook Creator")

text_rewriter = pipeline("text-generation", model="huggingface/granite")  # placeholder

# Rewrite text (English)
rewritten_text = text_rewriter(f"Rewrite this in {tone} tone:\n{text_content}", max_length=1500)[0]['generated_text']

# Translate to Telugu
telugu_text = translate_to_telugu(rewritten_text)

# Generate Telugu audio
tts_model.tts_to_file(text=telugu_text, speaker="alloy", file_path="telugu_output.mp3")
st.audio("telugu_output.mp3")
st.download_button("Download Audio", "telugu_output.mp3")

tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

uploaded_file = st.file_uploader(
    "Upload a document (TXT, DOCX, PDF)", 
    type=["txt", "docx", "pdf"]
)

text_input = st.text_area("Or paste your text here:")

text = ""

if uploaded_file is not None:
    try:
        text = read_file(uploaded_file)
        st.success(f"Loaded {uploaded_file.name} ✅")
    except Exception as e:
        st.error(f"Error reading file: {e}")

elif text_input.strip():
    text = text_input

else:
    st.info("Please upload a file or paste some text.")

# Load environment variables from .env file
try:
    from load_env import load_env_file
    load_env_file()
except ImportError:
    print("load_env.py not found, using system environment variables")

from utils import read_txt
from rewrite import rewrite_with_tone
from tts import synthesize_speech

# Streamlit Page Config
st.set_page_config(page_title="EchoVerse - AI Audiobook Creator", layout="wide")

# Initialize session state
if 'rewritten_text' not in st.session_state:
    st.session_state.rewritten_text = ""
if 'original_text' not in st.session_state:
    st.session_state.original_text = ""

st.title("🎧 EchoVerse - AI-Powered Audiobook Creator")
st.write("Convert your text into expressive audiobooks with customizable tones and voices.")

# --- Input Section ---
st.header("📝 Text Input")
uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
input_text = st.text_area("Or paste your text here:", height=200)

# Handle file upload
if uploaded_file is not None:
    try:
        input_text = read_txt(uploaded_file)
        st.success(f"✅ File uploaded successfully! ({len(input_text)} characters)")
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")

# Store input text in session state
if input_text:
    st.session_state.original_text = input_text

# --- Tone Selection ---
st.header("🎭 Tone Selection")
tone_options = {
    "Neutral": "📖 Clear, balanced, and informative",
    "Suspenseful": "🔍 Dramatic, mysterious, and tension-building", 
    "Inspiring": "⭐ Uplifting, motivational, and empowering"
}

tone = st.radio(
    "Choose a narration tone:",
    list(tone_options.keys()),
    format_func=lambda x: f"{x} - {tone_options[x]}"
)

# --- Action Buttons ---
col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Rewrite Text", type="secondary", use_container_width=True):
        if not input_text.strip():
            st.warning("⚠️ Please provide some text before rewriting.")
        else:
            try:
                with st.spinner(f"🤖 Rewriting text with {tone} tone..."):
                    st.session_state.rewritten_text = rewrite_with_tone(input_text, tone)
                st.success(f"✅ Text rewritten successfully with {tone} tone!")
            except Exception as e:
                st.error(f"❌ Error rewriting text: {str(e)}")
                st.info("💡 Make sure your Hugging Face token is set in environment variables")

with col2:
    if st.button("🎵 Generate Audio", type="primary", use_container_width=True, 
                disabled=not st.session_state.rewritten_text):
        if st.session_state.rewritten_text:
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    output_file = tmp_file.name
                
                with st.spinner("🎤 Generating speech... This may take a few minutes."):
                    synthesize_speech(st.session_state.rewritten_text, output_file)
                
                # Read and display audio
                if os.path.exists(output_file):
                    with open(output_file, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                    
                    st.success("✅ Audio generated successfully!")
                    
                    # Audio player and download
                    st.header("🎧 Your Audiobook")
                    st.audio(audio_bytes, format="audio/wav")
                    st.download_button(
                        "⬇️ Download Audio",
                        audio_bytes,
                        file_name=f"echoverse_{tone.lower()}_audiobook.wav",
                        mime="audio/wav"
                    )
                    
                    # Clean up temporary file
                    os.unlink(output_file)
                else:
                    st.error("❌ Audio file was not created")
                    
            except Exception as e:
                st.error(f"❌ Error generating audio: {str(e)}")
                st.info("💡 Make sure TTS dependencies are installed correctly")
        else:
            st.warning("⚠️ Please rewrite the text first before generating audio.")

# --- Text Comparison ---
if st.session_state.original_text and st.session_state.rewritten_text:
    st.header("📊 Text Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📄 Original Text")
        st.text_area("", value=st.session_state.original_text, height=300, disabled=True, key="orig")
        
        # Stats
        orig_words = len(st.session_state.original_text.split())
        st.metric("Word Count", orig_words)
        
    with col2:
        st.subheader(f"✨ {tone} Version")
        st.text_area("", value=st.session_state.rewritten_text, height=300, disabled=True, key="rewr")
        
        # Stats
        rewr_words = len(st.session_state.rewritten_text.split())
        st.metric("Word Count", rewr_words)

# --- Sidebar with info ---
with st.sidebar:
    st.header("ℹ️ About EchoVerse")
    st.write("""
    **Features:**
    - 🤖 AI-powered text rewriting
    - 🎭 Multiple tone options
    - 🎤 Text-to-speech conversion
    - 📊 Side-by-side comparison
    - ⬇️ Downloadable audio files
    
    **Requirements:**
    - Hugging Face token (for AI models)
    - TTS package installation
    """)
    
    st.header("🔧 Setup Guide")
    st.code("""
# Set your HF token
export HF_TOKEN="your_token_here"

# Install TTS package
pip install TTS

# Run the app
streamlit run app.py
    """)
    
    if st.button("📄 Load Sample Text"):
        sample_text = """
        Artificial Intelligence represents one of the most significant technological advances of our time. From machine learning algorithms that can recognize patterns in vast datasets to natural language processing systems that can understand and generate human-like text, AI is transforming industries and reshaping our understanding of what machines can accomplish.

        The journey of AI development has been marked by breakthrough moments, each building upon previous discoveries. Today, we stand at the threshold of an era where AI systems can assist in creative tasks, scientific research, and complex problem-solving across numerous domains.
        """
        st.session_state.original_text = sample_text
        st.rerun()