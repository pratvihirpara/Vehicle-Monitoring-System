from fastapi import FastAPI, File, UploadFile, HTTPException # type: ignore
from fastapi.responses import FileResponse, JSONResponse # type: ignore
import shutil
import os
import asyncio
import subprocess
from pathlib import Path
from ultralytics.utils.downloads import attempt_download_asset # type: ignore

app = FastAPI()

UPLOAD_DIR = "d:/vechicle-master/traffic/uploads"
OUTPUT_DIR = "d:/vechicle-master/backend/outputs"
MODEL_PATH = "d:/vechicle-master/traffic/yolov8n.pt"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ensure the YOLOv8 model file is downloaded
attempt_download_asset(MODEL_PATH)

async def process_video(input_path, output_path):
    """Runs the traffic2.py script asynchronously."""
    try:
        process = await asyncio.create_subprocess_exec(
            "python", "d:/vechicle-master/traffic/traffic2.py", input_path, output_path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise Exception(f"Error processing video: {stderr.decode()}")
    except Exception as e:
        print(f"Video processing failed: {e}")

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Handles file upload and triggers processing."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    output_file_path = os.path.join(OUTPUT_DIR, "output.mp4")

    # Run the video processing asynchronously
    asyncio.create_task(process_video(file_path, output_file_path))
    
    return {"filename": file.filename, "output_file": "output.mp4"}

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Serves the processed video for download."""
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(file_path, media_type="video/mp4", filename=filename)

@app.get("/output/")
async def get_output_video():
    """Serves the processed video for preview."""
    file_path = os.path.join(OUTPUT_DIR, "output.mp4")
    
    # Debugging: Print file path
    print(f"Checking file path: {file_path}")  
    
    if not os.path.exists(file_path):
        print("Processed video not found!")  # Debugging
        return JSONResponse(status_code=404, content={"error": "Processed video not found"})
    
    print("Serving the processed video!")  # Debugging
    return FileResponse(file_path, media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
