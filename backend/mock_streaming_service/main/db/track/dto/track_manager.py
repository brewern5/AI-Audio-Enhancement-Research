import librosa #type: ignore
import os
import soundfile as sf #type: ignore


class TrackManager:
    def __init__(self, lossy_path, lossless_path):
        self.lossy_path = lossy_path
        self.lossless_path = lossless_path

    def _check_file_exists(self, path):
        pass

    def get_lossless_stream_url(self):
        return self._chunk_file(self.lossless_path, chunk_size=32768)

    def get_lossy_stream_url(self):
        return self._chunk_file(self.lossy_path)

    """
        Audio operations file
    """

    def _load_audio(self, path): 
        return librosa.load(path, sr=None) # sr = None will change the default sample rate from 22kHz to the file detected Sample Rate
    
    def _chunk_file(self, path, chunk_size=16384): # 16KB chunks
        with open(path, 'rb') as audio_file:
            while True:
                chunk = audio_file.read(chunk_size)
                if not chunk:
                    break
                yield chunk