import os
import soundfile as sf
from utils.io_util import load_audio

class AudioFile:
    def __init__(self, file_path):
        self.file_path = file_path
        self.audio_data, self.sample_rate = load_audio(self.file_path)
        self.title = os.path.basename(self.file_path)
        
        # Get file metadata
        self.file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        # Get audio properties using soundfile
        try:
            info = sf.info(file_path)
            self.bit_depth = info.subtype_info.split('_')[-1] if hasattr(info, 'subtype_info') else 'N/A'
            self.channels = info.channels
            # Calculate bitrate (bits per second)
            self.bitrate_kbps = (self.file_size_mb * 8 * 1024) / info.duration if info.duration > 0 else 0
        except:
            # Fallback if soundfile can't read the file
            self.bit_depth = 'N/A'
            self.channels = 1 if len(self.audio_data.shape) == 1 else self.audio_data.shape[0]
            self.bitrate_kbps = 0
    
    def get_metadata_string(self):
        """Return formatted metadata string for display"""
        return (f"Sample Rate: {self.sample_rate} Hz | "
                f"Bit Depth: {self.bit_depth} | "
                f"Bitrate: {self.bitrate_kbps:.1f} kbps | "
                f"File Size: {self.file_size_mb:.2f} MB")