import numpy as np
import librosa

                    #sr=none keeps the original sample rate
def read_mono(path, sample_rate=None):
    audio_signal, sampl_rate = librosa.load(path, sr=sample_rate, mono=True)
    return audio_signal, sample_rate

# @param: audio_signal is the processed audio file being passed
# @param; n_fft is "Fast Fourier Transform"
def compute_shift_magnitude(audio_signal, n_fft):
    return