import matplotlib.pyplot as plt
import numpy as np
import librosa

class Comparison:

    @staticmethod
    def align_signals(wav, mp3):
        """
        Align two signals using cross-correlation. Returns aligned AudioFile objects.
        Note: If your files are already aligned, you can skip calling this method.
        """
        # Extract audio data from AudioFile objects
        if hasattr(wav, 'audio_data'):
            wav_data = wav.audio_data.copy()  # Use copy to avoid modifying original
        else:
            wav_data = wav
            
        if hasattr(mp3, 'audio_data'):
            mp3_data = mp3.audio_data.copy()  # Use copy to avoid modifying original
        else:
            mp3_data = mp3
            
        min_len = min(len(wav_data), len(mp3_data))

        # Removes white-space and ensures lengths are the same
        wav_data = wav_data[:min_len]
        mp3_data = mp3_data[:min_len]

        # Use a downsample for faster cross-correlation
        sample_rate = wav.sample_rate if hasattr(wav, 'sample_rate') else 44100
        downsample_factor = 10  # Downsample for speed
        
        # Downsample for correlation
        wav_downsampled = wav_data[::downsample_factor]
        mp3_downsampled = mp3_data[::downsample_factor]
        
        # Use only first 5 seconds or available data
        window_size = min(int(5 * sample_rate / downsample_factor), len(wav_downsampled))
        
        wav_window = wav_downsampled[:window_size]
        mp3_window = mp3_downsampled[:window_size]
        
        # The cross-correlation on windowed, downsampled data
        corr = np.correlate(wav_window, mp3_window, mode='same')
        shift = (np.argmax(corr) - len(mp3_window) // 2) * downsample_factor  # Scale back up

        # Apply shift if significant (more than 0.1 seconds)
        if abs(shift) > sample_rate * 0.1:
            if shift > 0:
                # Wav is shifted forward
                wav_data = wav_data[shift:]
                mp3_data = mp3_data[:len(wav_data)]
            elif shift < 0:
                # mp3 is shifted forward
                mp3_data = mp3_data[-shift:]
                wav_data = wav_data[:len(mp3_data)]
            
            print(f"Applied alignment shift: {shift} samples ({shift/sample_rate:.3f} seconds)")
        else:
            print("No significant alignment needed")

        # Update the AudioFile objects with aligned data
        if hasattr(wav, 'audio_data'):
            wav.audio_data = wav_data
        if hasattr(mp3, 'audio_data'):
            mp3.audio_data = mp3_data

        # Returns the original AudioFile objects with updated data
        return wav, mp3

    @staticmethod
    def chroma_differential(wav_chunks, mp3_chunks):
        """
        Simple differential for chroma features (12 pitch classes).
        No frequency cutoff logic needed - just element-wise difference.
        """
        differential = []
        min_chunks = min(len(wav_chunks), len(mp3_chunks))

        for i in range(min_chunks):
            wav_chunk = wav_chunks[i]
            mp3_chunk = mp3_chunks[i]
            
            # Get minimum dimensions
            min_pitch_bins = min(wav_chunk.shape[0], mp3_chunk.shape[0])
            min_time_frames = min(wav_chunk.shape[1], mp3_chunk.shape[1])
            
            # Trim to same size
            wav_trimmed = wav_chunk[:min_pitch_bins, :min_time_frames]
            mp3_trimmed = mp3_chunk[:min_pitch_bins, :min_time_frames]
            
            # Simple difference for chroma
            diff_chunk = np.subtract(wav_trimmed, mp3_trimmed)
            differential.append(diff_chunk)
        
        return differential

    @staticmethod
    def spectrogram_differential(wav_chunks, mp3_chunks, wav_file=None, mp3_file=None):
        """
        Calculate the difference between WAV and MP3 spectrograms with proper time alignment.
        Accounts for different sample rates. MP3 frequencies beyond its Nyquist limit 
        are padded with -80 dB (silence) to show they don't exist in MP3.
        """
        import scipy.ndimage
        
        differential = []
        min_chunks = min(len(wav_chunks), len(mp3_chunks))

        for i in range(min_chunks):
            wav_chunk = wav_chunks[i]
            mp3_chunk = mp3_chunks[i]
            
            # Handle frequency bins - pad MP3 with silence (-80 dB) for missing high frequencies
            if mp3_chunk.shape[0] < wav_chunk.shape[0]:
                # MP3 has fewer frequency bins - pad with -80 dB (represents no signal/silence)
                pad_amount = wav_chunk.shape[0] - mp3_chunk.shape[0]
                mp3_freq_padded = np.pad(mp3_chunk, ((0, pad_amount), (0, 0)), 
                                         mode='constant', constant_values=-80.0)
                wav_freq_trimmed = wav_chunk
            elif wav_chunk.shape[0] < mp3_chunk.shape[0]:
                # WAV has fewer bins - trim MP3 to match
                wav_freq_trimmed = wav_chunk
                mp3_freq_padded = mp3_chunk[:wav_chunk.shape[0], :]
            else:
                # Same number of frequency bins
                wav_freq_trimmed = wav_chunk
                mp3_freq_padded = mp3_chunk
            
            # If time frames don't match, interpolate to align them
            if wav_freq_trimmed.shape[1] != mp3_freq_padded.shape[1]:
                # Determine which needs interpolation
                if wav_freq_trimmed.shape[1] > mp3_freq_padded.shape[1]:
                    # Interpolate MP3 to match WAV time frames
                    zoom_factor = wav_freq_trimmed.shape[1] / mp3_freq_padded.shape[1]
                    mp3_freq_padded = scipy.ndimage.zoom(mp3_freq_padded, (1, zoom_factor), order=1)
                else:
                    # Interpolate WAV to match MP3 time frames
                    zoom_factor = mp3_freq_padded.shape[1] / wav_freq_trimmed.shape[1]
                    wav_freq_trimmed = scipy.ndimage.zoom(wav_freq_trimmed, (1, zoom_factor), order=1)
            
            # Now both should have the same shape - trim any rounding differences
            min_time_frames = min(wav_freq_trimmed.shape[1], mp3_freq_padded.shape[1])
            wav_trimmed = wav_freq_trimmed[:, :min_time_frames]
            mp3_trimmed = mp3_freq_padded[:, :min_time_frames]

            # Calculate full frequency range difference
            # Where MP3 was padded (-80 dB), the diff will show WAV - (-80), clearly showing missing content
            diff_chunk = np.subtract(wav_trimmed, mp3_trimmed)
            differential.append(diff_chunk)
            
        return differential
    
    @staticmethod
    def display_difference(chunks, reference_file, ax=None, feature_type: str='mel'):
        """
        Display the difference between two audio features (mel spectrogram or chroma).
        
        Args:
            chunks: List of difference chunks (numpy arrays)
            reference_file: AudioFile object with metadata
            ax: Matplotlib axes object (optional)
            feature_type: 'mel' for spectrogram or 'chroma' for chroma features
        """
        if not chunks:
            print("No differential Chunks to display")
            return
        
        # Concatenate all chunks into a single array
        full_difference = np.concatenate(chunks, axis=1)
        
        # Create figure if ax is not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            show_plot = True
        else:
            fig = ax.figure
            show_plot = False
        
        # Display based on feature type
        if feature_type == 'mel':
            # Calculate hop_length used in preprocessing (time-based)
            hop_length = int(0.0116 * reference_file.sample_rate)
            
            img = librosa.display.specshow(
                full_difference,
                x_axis='time',
                y_axis='mel',
                sr=reference_file.sample_rate,
                hop_length=hop_length,
                ax=ax,
                cmap='RdBu_r'  # Red-Blue colormap for differences
            )
            ax.set_title('Mel Spectrogram Difference', pad=15)
        elif feature_type == 'chroma':
            # Calculate hop_length used in preprocessing (time-based)
            hop_length = int(0.0116 * reference_file.sample_rate)
            
            img = librosa.display.specshow(
                full_difference,
                x_axis='time',
                y_axis='chroma',
                sr=reference_file.sample_rate,
                hop_length=hop_length,
                ax=ax,
                cmap='RdBu_r'  # Red-Blue colormap for differences
            )
            ax.set_title('Chroma Features Difference', pad=15)
        else:
            # Default display
            hop_length = int(0.0116 * reference_file.sample_rate)
            
            img = librosa.display.specshow(
                full_difference,
                x_axis='time',
                y_axis='linear',
                sr=reference_file.sample_rate,
                hop_length=hop_length,
                ax=ax,
                cmap='RdBu_r'
            )
            ax.set_title(f'{feature_type.capitalize()} Difference', pad=15)
        
        # Add colorbar
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        
        if show_plot:
            plt.show()
            