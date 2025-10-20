import os
import pandas as pd

# --- Configuration ---
# Define the paths to your data directories in Kaggle
wav_dir = "../data/SmallDataSet"
mp3_128k_dir = "../data/KaggleDirectories/mp3_converted"
mp3_192k_dir = "../data/KaggleDirectories/mp3_converted_192k"

# Output path for the final CSV file
output_csv_path = "../data/KaggleDirectories/dataset_manifest.csv"

# --- Logic to Create Pairs ---
data_pairs = []

# We'll use the original WAV files as the source of truth
print(f"Scanning for WAV files in: {wav_dir}")
for wav_filename in os.listdir(wav_dir):
    if wav_filename.lower().endswith(".wav"):
        base_name = os.path.splitext(wav_filename)[0]
        wav_path = os.path.join(wav_dir, wav_filename)

        # --- Pair with 128k MP3 ---
        mp3_128k_path = os.path.join(mp3_128k_dir, f"{base_name}.mp3")
        if os.path.exists(mp3_128k_path):
            data_pairs.append({
                "input_path": mp3_128k_path,
                "target_path": wav_path,
                "bitrate_kbps": 128,
                "original_id": base_name
            })
        else:
            print(f"Warning: Could not find matching 128k MP3 for {wav_filename}")

        # --- Pair with 192k MP3 ---
        mp3_192k_path = os.path.join(mp3_192k_dir, f"{base_name}.mp3")
        if os.path.exists(mp3_192k_path):
            data_pairs.append({
                "input_path": mp3_192k_path,
                "target_path": wav_path,
                "bitrate_kbps": 192,
                "original_id": base_name
            })
        else:
            print(f"Warning: Could not find matching 192k MP3 for {wav_filename}")

# --- Create and Save the DataFrame ---
if data_pairs:
    df = pd.DataFrame(data_pairs)
    df.to_csv(output_csv_path, index=False)
    print(f"\nSuccessfully created manifest file with {len(df)} pairs.")
    print(f"CSV saved to: {output_csv_path}")
    
    # Display the first few rows of the created table
    print("\n--- CSV Preview ---")
    print(df.head())
else:
    print("\nNo data pairs were created. Please check your directory paths.")