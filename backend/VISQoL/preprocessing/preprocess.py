import librosa

class Processor:
    def __init__(self):
        self = self

    # This prepares the audio to be sent to the VISQOL evaluation
    # A default sample rate must be set (48kHz)
    #
    def preprocess(self, waveform, sample_rate):
        