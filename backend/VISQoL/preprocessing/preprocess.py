import librosa

class Processor:
    def __init__(self):
        self = self

    # This prepares the audio to be sent to the VISQOL evaluation
    # A default sample rate must be set (48kHz) - This is a requirement by VISQOL
    # Mix-down to a mono single - This is a requirement by VISQOL
    # Vector Regression with a Max range of ~4.75 (?)
    def preprocess(self, waveform, sample_rate):
        new_waveform = librosa.resample(
            waveform, 
            orig_sr = sample_rate, 
            target_sr = 480000,     # The new sample rate we want
            res_type = "soxr_vhq",  # The preferred ressample type - sxor_vhq very high quality
        )

        return new_waveform
