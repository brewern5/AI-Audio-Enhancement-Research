import librosa
import matplotlib.pyplot as plt

from utils.audio_factory import AudioFactory

from preprocessing.preprocess import Processor # Prepare audio for evalutaion metrics
from output.spectrogram import Spectrogram 


def main():
    wav_audio_path = "./assets/example.wav"
    mp3_audio_path = "./assets/example.mp3"

    wav_audio_file = AudioFactory.create_audio_file(wav_audio_path)
    mp3_audio_file = AudioFactory.create_audio_file(mp3_audio_path)

    processor = Processor()

    wav_spectrogram_chunks = processor.get_spectrogram_features(wav_audio_file)
    mp3_spectrogram_chunks = processor.get_spectrogram_features(mp3_audio_file)

    wav_chroma_chunks = processor.get_chroma_chunks(wav_audio_file)
    mp3_chroma_chunks = processor.get_chroma_chunks(mp3_audio_file)

    fig, ((ax1, ax2), (ax3, ax4) )= plt.subplots(2, 2, figsize = (16, 12))

    Spectrogram.display_chunked_spectrogram(chunks=wav_spectrogram_chunks, file=wav_audio_file, ax = ax1)
    ax1.set_title(f'Spectrogram - {wav_audio_file.title}')

    Spectrogram.display_chunked_spectrogram(chunks=mp3_spectrogram_chunks, file=mp3_audio_file, ax = ax2)
    ax2.set_title(f'Spectrogram - {mp3_audio_file.title}')

    Spectrogram.display_chroma(wav_chroma_chunks, wav_audio_file, ax3)
    Spectrogram.display_chroma(mp3_chroma_chunks, mp3_audio_file, ax4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()