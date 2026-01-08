# EchoVerse


General Description
EchoVerse is an AI-powered audiobook creation system that transforms written text into expressive, downloadable audio narrations. The tool accepts either pasted text or uploaded .txt files and allows users to select a desired narration tone: Neutral, Suspenseful, or Inspiring.

The workflow involves:

Tone-Adaptive Rewriting of the input text using IBM Watsonx Granite LLM, which ensures that the rewritten text reflects the chosen style while preserving meaning.

Natural-Sounding Narration generated with IBM Watson Text-to-Speech (TTS), offering multiple voice options (e.g., Lisa, Michael, Allison).

Dual Text View showing the original and rewritten versions side by side for user validation.

Audio Output Options to either stream narration within the app or download it as .mp3 for offline use.

Streamlit -based Interface for a smooth, user-friendly experience.

How it solves the challenge:

Addresses accessibility needs, especially for visually impaired users.

Reduces time for students and professionals who prefer listening over reading.

Enables reusability of written material (study notes, reports, stories) in a more engaging format.

Provides tone adaptability, making the same content usable in academic, professional, or creative settings.

Novelty / Uniqueness:

Unlike standard text-to-speech tools, EchoVerse rewrites text into tone-specific styles before generating audio. This makes the narration more expressive and engaging.

Integrates prompt chaining with IBM Watsonx Granite LLM, ensuring the rewritten version is faithful to the original while adding stylistic richness.

Provides side-by-side text comparison, a feature not commonly found in audiobook generators, allowing users to trust the AI’s rewriting process.

Customizable voices + tones create a near-human audiobook experience tailored to audience preferences.

The local Streamlit-based app ensures simplicity and offline usability for individual users, without complex setup.

Business / Social Impact:

Business Impact:

Expands opportunities for publishers, educators, and corporates to repurpose existing content into audiobooks with minimal effort.

Reduces production costs for audiobook creation by automating narration.

Increases engagement for digital platforms (e.g., e-learning, knowledge-sharing apps).

Social Impact:

Makes written content more accessible to visually impaired users.

Helps students and working professionals consume information faster and more effectively through audio learning.

Encourages inclusivity by enabling content consumption across diverse audiences.

Provides a platform for creators and independent writers to easily publish audiobooks without professional recording setups.

Technology Architecture:

Business Impact:

Expands opportunities for publishers, educators, and corporates to repurpose existing content into audiobooks with minimal effort.

Reduces production costs for audiobook creation by automating narration.

Increases engagement for digital platforms (e.g., e-learning, knowledge-sharing apps).

Social Impact:

Makes written content more accessible to visually impaired users.

Helps students and working professionals consume information faster and more effectively through audio learning.

Encourages inclusivity by enabling content consumption across diverse audiences.

Provides a platform for creators and independent writers to easily publish audiobooks without professional recording setups.

Scope of the Work:

Phase 1: Core Functionality

Implement text input & .txt file upload.

Integrate Watsonx Granite for tone-specific rewriting.

Add side-by-side comparison display.

Phase 2: Audio Narration

Integrate Watson TTS for voice generation.

Provide multiple voice options and real-time playback.

Enable .mp3 download functionality.

Phase 3: User Experience Enhancement

Develop clean, accessible Streamlit interface.

Add support for larger documents (batch narration).

Ensure responsive design for web and mobile use.

Phase 4: Scalability & Future Extensions

Cloud deployment for multi-user access.

Expand voice/tone library (e.g., motivational, storytelling).

Add support for additional languages for global reach.
