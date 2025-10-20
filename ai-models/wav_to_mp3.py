import os
from pydub import AudioSegment

#Two versions of audio created
input_dir = "/kaggle/input/smalldataset" 
output_dir = "/kaggle/working/mp3_converted"
#output_dir = "/kaggle/working/mp3_converted_192k"

# Desired MP3 bitrate (e.g., '128k', '192k', '320k')
bitrate = "128k"
#bitrate = "192k"

# --- Conversion Logic ---
print(f"Starting conversion of WAV files from: {input_dir}")

# 1. Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# 2. List all files in the input directory
try:
    files = os.listdir(input_dir)
except FileNotFoundError:
    print(f"ERROR: Input directory not found: {input_dir}")
    print("Please make sure you have attached your dataset to the notebook and the path is correct.")
    files = []

converted_count = 0
# 3. Loop through files and convert the .wav files
for filename in files:
    if filename.lower().endswith(".wav"):
        try:
            # Construct full file paths
            wav_path = os.path.join(input_dir, filename)
            mp3_filename = os.path.splitext(filename)[0] + ".mp3"
            mp3_path = os.path.join(output_dir, mp3_filename)
            
            # Load the WAV file
            audio = AudioSegment.from_wav(wav_path)
            
            # Export as MP3 with specified bitrate
            print(f"Converting {filename} to MP3...")
            audio.export(mp3_path, format="mp3", bitrate=bitrate)
            converted_count += 1
            
        except Exception as e:
            print(f"Could not convert {filename}. Error: {e}")

print(f"\nConversion complete. Converted {converted_count} files.")
print(f"MP3 files are saved in: {output_dir}")