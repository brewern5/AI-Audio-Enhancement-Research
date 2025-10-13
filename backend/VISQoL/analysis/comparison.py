import matplotlib.pyplot as plt
import numpy as np
import librosa

class Comparison:

    # Passing the chunks of both the mp3 and wav files (both chroma and spectrogram)
    # return a new map of the differential chunks to be displayed on a new graph
    @staticmethod
    def differential(wav_chunks, mp3_chunks):

        differential = []

        min_chunks = min(len(wav_chunks), len(mp3_chunks))

        for i in range(min_chunks):
            wav_chunk = wav_chunks[i]
            mp3_chunk = mp3_chunks[i]

            # Taking min dimension of the different chunk sizes
            min_freq_bins = min(wav_chunk.shape[0], mp3_chunk.shape[0])
            min_time_frames = min(wav_chunk.shape[1], mp3_chunk.shape[1])

            # Trim to same size
            wav_trimmed = wav_chunk[:min_freq_bins, :min_time_frames]
            mp3_trimmed = mp3_chunk[:min_freq_bins, :min_time_frames]

            # Get difference
            diff_chunk = np.subtract(wav_trimmed, mp3_trimmed)
            differential.append(diff_chunk)

        return differential
    
    @staticmethod
    def display_difference(chunks, reference_file, ax=None, feature_type: str = 'mel'):
        if not chunks:
            print("No differential Chunks to display")
            
        full_difference = np.concatenate(chunks, axis=1)

        if ax is None:
            fig, ax = plt.subplots(figsize = (12, 6))
            show_plot = True
        else:
            fig = ax.figure
            show_plot = False

        img = librosa.display.specshow(
            full_difference,
            x_axis = 'time',
            y_axis = feature_type,
            sr = reference_file.sample_rate,
            ax = ax
        )

        ax.set_title(f'Differential for: {feature_type}')
        fig.colorbar(img, ax = ax, format = "%+2.f")

        if show_plot:
            plt.show()