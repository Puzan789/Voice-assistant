import os
import wave
import numpy as np
import pyaudio
import whisper
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from scipy.io import wavfile
from gtts import gTTS
import pygame
from langchain.prompts import ChatPromptTemplate

# Load Whisper model for speech-to-text
def load_whisper():
    model = whisper.load_model("base")
    return model

# Silence detection
def is_silence(data, max_amplitude_threshold=3000):
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold

# Record audio
def record_audio_chunk(audio, stream, chunk_length=5):
    print("Recording...")
    frames = []
    num_chunks = int(16000 / 1024 * chunk_length)
    for _ in range(num_chunks):
        data = stream.read(1024)
        frames.append(data)

    temp_file_path = './temp_audio_chunk.wav'
    print("Writing...")
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    try:
        samplerate, data = wavfile.read(temp_file_path)
        if is_silence(data):
            os.remove(temp_file_path)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error while reading audio file: {e}")

# Transcribe audio to text
def transcribe_audio(model, file_path):
    print("Transcribing...")
    if os.path.isfile(file_path):
        results = model.transcribe(file_path)
        return results['text']
    else:
        return None

# Load the LLM model for both intent classification and response generation
def load_llm():
    chat_groq = ChatGroq(temperature=0, model_name="llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
    return chat_groq

# Use the LLM to classify the intent of the user's input





def classify_intent_with_llm(user_question):
    llm=load_llm()
    valid_intents = [
        "order_status",
        "return_policy",
        "refund_status",
        "payment_issue",
        "product_inquiry",
        "general_inquiry"
    ]
    
    # Create the options for the intents
    intent_options = "\n".join(f"- {intent}" for intent in valid_intents)
    
    # Generate the prompt
    prompt = f"""
    You are an AI assistant. Your job is to classify the user's input into one of the following intents. Return only the intent, without any additional words:

    {intent_options}

    User: "{user_question}"
    Intent:
    """
    prompt=ChatPromptTemplate.from_template(prompt)
    prompt=prompt.invoke({"user_input":user_question})
    result=llm.invoke(prompt)
    
    return result.content




def get_response_llm(user_question, intent):
    input_prompt = load_general_prompt(user_question, intent)

    chat_groq = load_llm()
    prompt = PromptTemplate.from_template(input_prompt)

    input_data = {
        "question": user_question,
        "intent": intent,
    }

    chain = LLMChain(
        llm=chat_groq,
        prompt=prompt,
        verbose=True,
    )

    response = chain.invoke(input_data)
    return response['text']



def load_general_prompt(user_question, intent):
    input_prompt = f"""
    As an expert customer service representative, your goal is to provide accurate and concise answers to customer inquiries.

    If the customer's intent is clear and related to one of the following categories: {intent}, provide a specific response based on that intent.
    If the intent is not clear or does not match any specific category, provide a general helpful response.

    Customer question: {user_question}

    Response:
    """
    return input_prompt

# Play text as speech
def play_text_to_speech(text, language='en', slow=False):
    tts = gTTS(text=text, lang=language, slow=slow)
    temp_audio_file = "temp_audio.mp3"
    tts.save(temp_audio_file)
    pygame.mixer.init()
    pygame.mixer.music.load(temp_audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    os.remove(temp_audio_file)
