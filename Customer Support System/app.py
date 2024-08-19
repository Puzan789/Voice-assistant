import os
import pyaudio
import streamlit as st
from utils import record_audio_chunk, transcribe_audio, get_response_llm, play_text_to_speech, load_whisper, classify_intent_with_llm, load_llm

# Initialize Whisper model for speech-to-text
model = load_whisper()

# Load the LLM for intent classification and response generation
llm = load_llm()

chunk_file = 'temp_audio_chunk.wav'

def main():
    st.title("Customer Support System")

    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display the conversation history
    for message in st.session_state.chat_history:
        if message['type'] == 'user':
            st.text(f"You: {message['text']}")
        else:
            st.text(f"AI: {message['text']}")

    # Audio input section
    if st.button("Start Recording"):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        
        try:
            if record_audio_chunk(audio, stream):
                st.warning("Silence detected, no audio recorded.")
            else:
                text = transcribe_audio(model, chunk_file)

                if text:
                    st.session_state.chat_history.append({'type': 'user', 'text': text})
                    intent = classify_intent_with_llm(text)
                    response_llm = get_response_llm(user_question=text, intent=intent)
                    st.session_state.chat_history.append({'type': 'ai', 'text': response_llm})
                    play_text_to_speech(text=response_llm)
                    os.remove(chunk_file)
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
        st.experimental_rerun()  

    # Text input section
    user_prompt = st.text_input("Type your message here...")
    if st.button("Submit") and user_prompt:  # Add a submit button for text input
        st.session_state.chat_history.append({'type': 'user', 'text': user_prompt})
        intent = classify_intent_with_llm(user_prompt)
        response = get_response_llm(user_question=user_prompt, intent=intent)
        st.session_state.chat_history.append({'type': 'ai', 'text': response})
        st.experimental_rerun()  # Rerun after processing text input to update UI

if __name__ == "__main__":
    main()
