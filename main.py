import os
import shutil
import json
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import whisper

# Load Whisper model
model = whisper.load_model("base")  # You can use "tiny", "base", "small", "medium", or "large"

#app initialization
app = FastAPI()

UPLOAD_FOLDER = "audio_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        print(f"Saving file to: {filepath}")
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("File saved. Now transcribing...")

        result = model.transcribe(filepath)
        transcription = result["text"].strip()

        # Save transcription to JSON file
        json_output_file = "transcriptions.json"
        if os.path.exists(json_output_file):
            with open(json_output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []

        data.append({
            "filename": filename,
            "transcription": transcription
        })

        with open(json_output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return {"filename": filename, "transcription": transcription}

    except Exception as e:
        print(f"Error occurred: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def root():
    return {"message": "Welcome to Whisper Transcription API"}
