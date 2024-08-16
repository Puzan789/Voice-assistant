import os
import pyaudio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from utils import record_audio_chunk, transcribe_audio, get_response_llm, play_text_to_speech, load_whisper, classify_intent_with_llm, load_llm
from langchain.memory import ConversationBufferMemory

app = FastAPI()

# Initialize Whisper model for speech-to-text
model = load_whisper()

# Load the LLM for intent classification and response generation
llm = load_llm()

memory = ConversationBufferMemory(memory_key="chat_history")

chunk_file = 'temp_audio_chunk.wav'

@app.post("/record-audio/")
async def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    
    try:
        if record_audio_chunk(audio, stream):
            return JSONResponse(content={"message": "Silence detected, no audio recorded."}, status_code=200)
        else:
            text = transcribe_audio(model, chunk_file)

            if text:
                memory.save_context({"user": text}, {})  # Save user input to memory
                intent = classify_intent_with_llm(text, llm)
                response_llm = get_response_llm(user_question=text, memory=memory, intent=intent)
                memory.save_context({}, {"ai": response_llm})  # Save AI response to memory
                play_text_to_speech(text=response_llm)
                os.remove(chunk_file)
                return JSONResponse(content={"user_input": text, "ai_response": response_llm}, status_code=200)
            else:
                raise HTTPException(status_code=500, detail="Transcription failed, please try again.")

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    file_location = f"./{chunk_file}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    text = transcribe_audio(model, file_location)
    if text:
        memory.save_context({"user": text}, {})  # Save user input to memory
        intent = classify_intent_with_llm(text, llm)
        response_llm = get_response_llm(user_question=text, memory=memory, intent=intent)
        memory.save_context({}, {"ai": response_llm})  # Save AI response to memory
        play_text_to_speech(text=response_llm)
        os.remove(file_location)
        return JSONResponse(content={"user_input": text, "ai_response": response_llm}, status_code=200)
    else:
        raise HTTPException(status_code=500, detail="Transcription failed, please try again.")


@app.post("/get-response/")
async def get_response(user_prompt: str):
    memory.save_context({"user": user_prompt}, {})  # Save user input to memory
    intent = classify_intent_with_llm(user_prompt, llm)
    response = get_response_llm(user_question=user_prompt, memory=memory, intent=intent)
    memory.save_context({}, {"ai": response})  # Save AI response to memory
    return JSONResponse(content={"ai_response": response}, status_code=200)


@app.get("/chat-history/")
async def get_chat_history():
    return JSONResponse(content=memory.load_memory(), status_code=200)

@app.post("/tts/")
async def text_to_speech(text: str):
    play_text_to_speech(text=text)
    return JSONResponse(content={"message": "TTS played successfully."}, status_code=200)

@app.get("/download-tts/")
async def download_tts(text: str):
    play_text_to_speech(text=text)
    temp_audio_file = "temp_audio.mp3"
    if os.path.exists(temp_audio_file):
        return FileResponse(path=temp_audio_file, filename=temp_audio_file)
    else:
        raise HTTPException(status_code=404, detail="Audio file not found.")
