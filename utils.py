import os 
from dotenv import load_dotenv
import wave 
import pyaudio
from scipy.io import wavfile
import numpy as np 
import whisper 
from langchain.chains.llms import LLMChain 
from langchain_core.prompts import PromptTemplate
from langchain_groq import chat_groq 
from gtts import gTTS
load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")


def is_silence(data,max_amplitude_threshold=3000):
    """Check if audio data contains silence."""
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold


