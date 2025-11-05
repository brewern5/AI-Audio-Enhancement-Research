import matplotlib.pyplot as plt

from utils.audio_factory import AudioFactory

from preprocessing.preprocess import Processor # Prepare audio for evalutaion metrics
from output.spectrogram import Spectrogram 
from analysis.comparison import Comparison


def main():
    wav_audio_path = "./assets/1000hz_sine_48_og.wav"
    mp3_audio_path = "./assets/1000hz_sine_44.mp3"

    # Load audio file
    wav_audio_file = AudioFactory.create_audio_file(wav_audio_path)
    mp3_audio_file = AudioFactory.create_audio_file(mp3_audio_path)

    wav_audio_file, mp3_audio_file = Comparison.align_signals(wav_audio_file, mp3_audio_file)

    processor = Processor()

    # Get Spectrograms
    wav_spectrogram_chunks = processor.get_spectrogram_features(wav_audio_file)
    mp3_spectrogram_chunks = processor.get_spectrogram_features(mp3_audio_file)

    # Get Chromas
    wav_chroma_chunks = processor.get_chroma_chunks(wav_audio_file)
    mp3_chroma_chunks = processor.get_chroma_chunks(mp3_audio_file)

    # Get difference with both chroma and spectrogram
    spectrogram_diff = Comparison.spectrogram_differential(wav_spectrogram_chunks, mp3_spectrogram_chunks)
    chromas_diff = Comparison.chroma_differential(wav_chroma_chunks, mp3_chroma_chunks)

    # Create first display for Mel Spectrograms
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 18),
                                         gridspec_kw={'hspace':0.6})
    #plt.subplots_adjust(hspace=0.4)
    
    Spectrogram.display_chunked_spectrogram(chunks=wav_spectrogram_chunks, file=wav_audio_file, ax=ax1)
    #ax1.xaxis.set_label_position('top')
    #ax1.axis.tick_top()
    ax1.text(0.5, 1.02, wav_audio_file.get_metadata_string(), 
             transform=ax1.transAxes, ha='center', va='bottom', 
             fontsize=9, style='italic')

    Spectrogram.display_chunked_spectrogram(chunks=mp3_spectrogram_chunks, file=mp3_audio_file, ax=ax2)
    #ax2.xaxis.set_label_position('top')
    #ax2.axis.tick_top()
    ax2.text(0.5, 1.02, mp3_audio_file.get_metadata_string(), 
             transform=ax2.transAxes, ha='center', va='bottom',
             fontsize=9, style='italic')
    
    #ax3.xaxis.set_label_position('top')
    Comparison.display_difference(chunks=spectrogram_diff, reference_file=wav_audio_file, ax=ax3, feature_type='mel')
    
    fig1.suptitle('Mel Spectrogram Analysis', fontsize=16, fontweight='bold', y=0.995)

    # Create second display for Chroma Features
    fig2, (ax4, ax5, ax6) = plt.subplots(3, 1, figsize=(14, 18),
                                         gridspec_kw={'hspace':0.6})
    #plt.subplots_adjust(hspace=0.4)
    
    Spectrogram.display_chroma(chunks=wav_chroma_chunks, file=wav_audio_file, ax=ax4)
    ax4.text(0.5, 1.02, wav_audio_file.get_metadata_string(), 
             ha='center', va='bottom', transform=ax4.transAxes, fontsize=9, style='italic')
    
    Spectrogram.display_chroma(chunks=mp3_chroma_chunks, file=mp3_audio_file, ax=ax5)
    ax5.text(0.5, 1.02, mp3_audio_file.get_metadata_string(), 
             ha='center', va='bottom', transform=ax5.transAxes, fontsize=9, style='italic')
    
    Comparison.display_difference(chunks=chromas_diff, reference_file=mp3_audio_file, ax=ax6, feature_type='chroma')
    
    fig2.suptitle('Chroma Features Analysis', fontsize=16, fontweight='bold', y=0.995)

    plt.show()


if __name__ == "__main__":
    main()