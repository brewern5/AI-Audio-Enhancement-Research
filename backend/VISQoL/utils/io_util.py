import librosa # Loading in audio files 

def load_audio( audio_path):
    
    # Converts the file into a waveform (list) and a sample rate
    return librosa.load(audio_path, sr=None) # sr = None will change the default sample rate from 22kHz to the file detected Sample Rate
