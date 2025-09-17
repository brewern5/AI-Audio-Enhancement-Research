import librosa

from utils.io_util import load_audio # Main I/O
from preprocessing.preprocess import Processor # Prepare audio for evalutaion metrics
from output.spectrogram import display_spectrogram 

def main():
    #audio_path = "./assets/example.wav"
    #waveform, sample_rate = load_audio(audio_path)
    #print("Waveform: ", waveform)
    #print("Sample Rate:", sample_rate)

    waveform, sample_rate = load_audio(librosa.ex("trumpet"))

    processor = Processor()

    #processor_return = processor.preprocess(waveform, sample_rate)

    #print("Processor Return: ", processor_return)
    
    display_spectrogram(waveform, sample_rate=sample_rate)


if __name__ == "__main__":
    main()