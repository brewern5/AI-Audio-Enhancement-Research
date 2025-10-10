import os
from utils.io_util import load_audio

class AudioFile:
    def __init__(self, file_path):
        self.file_path = file_path
        self.audio_data, self.sample_rate = load_audio(self.file_path)
        self.title = os.path.basename(self.file_path)