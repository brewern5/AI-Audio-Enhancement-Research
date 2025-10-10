import os
from .audio_file import AudioFile

class AudioFactory:

    def create_audio_file(file_path):

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Check the file extensions
        supported_extentsions = {'.wav', '.mp3'}

        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext not in supported_extentsions:
            raise ValueError(f"Unsupported audio format: {file_ext}")
        
        return AudioFile(file_path)