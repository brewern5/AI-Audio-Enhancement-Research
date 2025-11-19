import librosa #type: Ignore
import numpy as np

class Processor:
    def __init__(self):
        self = self

    # This prepares the audio to be sent to the VISQOL evaluation
    # Vector Regression with a Max range of ~4.75 (?)
    def preprocess(self, waveform, sample_rate):
        new_waveform = librosa.resample(
            waveform, 
            orig_sr = sample_rate, 
            target_sr = 960000,     # The new sample rate we want
            res_type = "soxr_vhq",  # The preferred ressample type - sxor_vhq very high quality
        )

        return new_waveform
    
    def _calclate_nfft(self, chunk_length, default_nfft = 2048):

        if chunk_length < default_nfft:
            n_fft = 2 ** int(np.log2(chunk_length))

            return max(n_fft, 256)
        return default_nfft

    # This method breaks larger files down into multiple parts for easier processing
    def _stream(self, file):

        # Use time-based parameters that scale with sample rate
        # Target ~46ms window (2048 samples at 44.1kHz)
        frame_length = int(0.046 * file.sample_rate)
        
        # Target ~11.6ms hop (512 samples at 44.1kHz = 75% overlap)
        hop_length = int(0.0116 * file.sample_rate)

        # Create a streaming iterator that reads the audio file in chunks
        # This prevents loading the entire file into memory at once
        # block_length=128 means we process 128 frames worth of audio per iteration
        return librosa.stream(
            file.file_path,
            block_length = 128,
            frame_length = frame_length,
            hop_length = hop_length
        )


    def get_chroma_chunks(self, file):
        
        # Use time-based parameters scaled to sample rate
        frame_length = int(0.046 * file.sample_rate)
        hop_length = int(0.0116 * file.sample_rate)

        stream = self._stream(file)

        # Initialize list to store chroma features from each audio chunk
        chromas = []

        # Process each chunk of audio from the stream
        for y in stream:

            # Calcuate n_fft for this chunk
            n_fft = self._calclate_nfft(len(y), frame_length)

            current_hop_length = min(hop_length, n_fft //4)
            # Extract chroma features (musical pitch information) from the current chunk
            # Chroma features represent the 12 different pitch classes (C, C#, D, etc.)
            chroma_block = librosa.feature.chroma_stft(
                y = y,                      # Audio chunk to analyze
                n_fft = n_fft,       # FFT window size (same as frame_length)
                hop_length = current_hop_length,    # Step size between analysis windows
                center = False              # Don't pad the signal (use raw chunk boundaries)
            )

            # Store the extracted chroma features for this chunk
            chromas.append(chroma_block)

        return chromas
    def get_spectrogram_features(self, file):
        # Use time-based parameters scaled to sample rate
        frame_length = int(0.046 * file.sample_rate)
        hop_length = int(0.0116 * file.sample_rate)

        stream = self._stream(file)

        spectrograms = []

        for i, chunk in enumerate(stream):

            n_fft = self._calclate_nfft(len(chunk), frame_length)

            current_hop_length = min(hop_length, n_fft //4)
            current_hop_length = min(hop_length, n_fft //4) 

            spectrogram_block = librosa.feature.melspectrogram(
                y = chunk,
                sr = file.sample_rate,
                n_fft = n_fft,
                hop_length = current_hop_length,
                center = False,
                fmax = 22050  # Limit to 22.05kHz max frequency for consistent comparison
            )

            # Convert to dB scale for visualization
            mel_spec_db = librosa.power_to_db(spectrogram_block, ref = np.max)

            spectrograms.append(mel_spec_db)

        return spectrograms