from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
import uvicorn

app = FastAPI(title="Audio Streaming Test Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define the base path for audio files
BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / "file_store"

@app.get("/")
async def root():
    return {"message": "Audio Streaming Test Server"}

@app.get("/tracks/list")
async def list_tracks():
    """List available tracks"""
    tracks = []
    
    # Check mp3 files
    mp3_dir = AUDIO_DIR / "mp3"
    if mp3_dir.exists():
        for file in mp3_dir.glob("*.mp3"):
            tracks.append({
                "name": file.stem,
                "format": "mp3",
                "type": "lossy"
            })
    
    # Check wav files  
    wav_dir = AUDIO_DIR / "wav"
    if wav_dir.exists():
        for file in wav_dir.glob("*.wav"):
            tracks.append({
                "name": file.stem,
                "format": "wav", 
                "type": "lossless"
            })
    
    return {"tracks": tracks}

@app.get("/tracks/stream/lossy")
async def stream_lossy_track(name: str = Query(..., description="Name of the track")):
    """Stream lossy (MP3) audio"""
    mp3_file = AUDIO_DIR / "mp3" / f"{name}.mp3"
    
    if not mp3_file.exists():
        raise HTTPException(status_code=404, detail=f"Track '{name}' not found")
    
    return FileResponse(
        path=mp3_file,
        media_type="audio/mpeg",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Disposition": f"inline; filename={name}.mp3"
        }
    )

@app.get("/tracks/stream/lossless")  
async def stream_lossless_track(name: str = Query(..., description="Name of the track")):
    """Stream lossless (WAV) audio"""
    wav_file = AUDIO_DIR / "wav" / f"{name}.wav"
    
    if not wav_file.exists():
        # Fallback to MP3 if WAV doesn't exist
        mp3_file = AUDIO_DIR / "mp3" / f"{name}.mp3"
        if mp3_file.exists():
            return FileResponse(
                path=mp3_file,
                media_type="audio/mpeg",
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Disposition": f"inline; filename={name}.mp3"
                }
            )
        raise HTTPException(status_code=404, detail=f"Track '{name}' not found")
    
    return FileResponse(
        path=wav_file,
        media_type="audio/wav",
        headers={
            "Accept-Ranges": "bytes", 
            "Content-Disposition": f"inline; filename={name}.wav"
        }
    )

@app.get("/tracks/newest")
async def get_newest_tracks():
    """Get sample track metadata for testing"""
    return [
        {
            "_id": "test1",
            "name": "nate_data1",
            "length": "00:02:30",
            "lossless": {
                "sample_rate": "44100",
                "size": "2.5 MB", 
                "bit_depth": "16",
                "bit_rate": "1411"
            },
            "lossy": {
                "sample_rate": "44100",
                "size": "1.2 MB",
                "bit_depth": "16", 
                "bit_rate": "320"
            }
        },
        {
            "_id": "test2", 
            "name": "sine_48_og",
            "length": "00:01:45",
            "lossless": {
                "sample_rate": "48000",
                "size": "3.1 MB",
                "bit_depth": "24", 
                "bit_rate": "2304"
            },
            "lossy": {
                "sample_rate": "48000",
                "size": "1.5 MB", 
                "bit_depth": "16",
                "bit_rate": "320"
            }
        }
    ]

if __name__ == "__main__":
    print("Starting Audio Streaming Test Server...")
    print(f"Audio files directory: {AUDIO_DIR}")
    print(f"Available MP3 files: {list((AUDIO_DIR / 'mp3').glob('*.mp3')) if (AUDIO_DIR / 'mp3').exists() else 'No MP3 directory'}")
    print(f"Available WAV files: {list((AUDIO_DIR / 'wav').glob('*.wav')) if (AUDIO_DIR / 'wav').exists() else 'No WAV directory'}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)