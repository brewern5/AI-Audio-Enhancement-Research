from utils.io_util import load_audio # Main I/O
from preprocessing import Processor # Prepare audio for evalutaion metrics

def main():
    audio_path = "./assets/example.wav"
    waveform, sample_rate = load_audio(audio_path)
    print("Waveform: ", waveform)
    print("Sample Rate:", sample_rate)

    processor = Processor()

    processor.preprocess(waveform, sample_rate)


if __name__ == "__main__":
    main()