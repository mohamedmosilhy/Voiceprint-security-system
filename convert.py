from pydub import AudioSegment
import os

def convert_m4a_to_wav(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".m4a"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".wav")

            # Load M4A file using pydub
            audio = AudioSegment.from_file(input_path, format="m4a")

            # Export as WAV
            audio.export(output_path, format="wav")

            print(f"Converted {filename} to {os.path.basename(output_path)}")

if __name__ == "__main__":
    # Set the input and output folders
    input_folder = "Members\Osama"
    output_folder = "Members\Osama"

    convert_m4a_to_wav(input_folder, output_folder)
