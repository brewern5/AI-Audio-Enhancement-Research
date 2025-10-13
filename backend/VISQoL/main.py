import matplotlib.pyplot as plt

from utils.audio_factory import AudioFactory

from preprocessing.preprocess import Processor # Prepare audio for evalutaion metrics
from output.spectrogram import Spectrogram 
from analysis.comparison import Comparison


def main():
    wav_audio_path = "./assets/example.wav"
    mp3_audio_path = "./assets/example.mp3"

    # Load audio file
    wav_audio_file = AudioFactory.create_audio_file(wav_audio_path)
    mp3_audio_file = AudioFactory.create_audio_file(mp3_audio_path)

    processor = Processor()

    # Get Spectrograms
    wav_spectrogram_chunks = processor.get_spectrogram_features(wav_audio_file)
    mp3_spectrogram_chunks = processor.get_spectrogram_features(mp3_audio_file)

    # Get Chromas
    wav_chroma_chunks = processor.get_chroma_chunks(wav_audio_file)
    mp3_chroma_chunks = processor.get_chroma_chunks(mp3_audio_file)

    # Get difference with both chroma and spectrogram
    spectrogram_diff = Comparison.differential(wav_spectrogram_chunks, mp3_spectrogram_chunks)
    chromas_diff = Comparison.differential(wav_chroma_chunks, mp3_chroma_chunks)

    # Create the main display
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6) )= plt.subplots(3, 2, figsize = (16, 18))

    Spectrogram.display_chunked_spectrogram(chunks=wav_spectrogram_chunks, file=wav_audio_file, ax = ax1)
    Spectrogram.display_chunked_spectrogram(chunks=mp3_spectrogram_chunks, file=mp3_audio_file, ax = ax3)

    Spectrogram.display_chroma(chunks=wav_chroma_chunks, file=wav_audio_file, ax=ax2)
    Spectrogram.display_chroma(chunks=mp3_chroma_chunks, file=mp3_audio_file, ax=ax4)

    
    Comparison.display_difference(chunks = spectrogram_diff, reference_file = wav_audio_file, ax = ax5, feature_type = 'mel')
    Comparison.display_difference(chunks = chromas_diff, reference_file = mp3_audio_file, ax = ax6, feature_type = 'chroma')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()