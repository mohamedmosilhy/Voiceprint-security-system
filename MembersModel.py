import os
import numpy as np
from pyAudioAnalysis import audioTrainTest as aT
import wave

class AccessModel:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.audio_data = self.load_audio_folder()

    def load_audio_folder(self):
        audio_data = {}
        for subfolder in os.listdir(self.folder_path):
            subfolder_path = os.path.join(self.folder_path, subfolder)
            if os.path.isdir(subfolder_path):
                for filename in os.listdir(subfolder_path):
                    if filename.endswith(".wav"):
                        file_path = os.path.join(subfolder_path, filename)
                        data, sample_rate = self.load_audio(file_path)
                        audio_data[subfolder] = (data, sample_rate)
        return audio_data

    def load_audio(self, filename):
        with wave.open(filename, 'rb') as wav_file:
            frames = wav_file.readframes(-1)
            data = np.frombuffer(frames, dtype=np.int16)
            sample_rate = wav_file.getframerate()
        return data, sample_rate

    def train_speaker_recognition_model(self):
        # Get a list of all speakers
        speakers = list(self.audio_data.keys())
        
        # Generate file paths for each speaker
        paths = [os.path.join(self.folder_path, speaker) for speaker in speakers]

        # Extract features from audio files and train the speaker recognition model
        aT.extract_features_and_train(
            paths,  # List of file paths
            1.0,    # Window size for feature extraction
            1.0,    # Step size for feature extraction
            aT.shortTermWindow,  # Short-term window size for feature extraction
            aT.shortTermStep,    # Short-term step size for feature extraction
            "svm",  # Model type
            "svm_Persons_model",  # Model file path
            False   # Flag to disable training visualization
        )

    
    def get_prediction(self, audio_file_path, model_file_path):
        """
        Get the prediction of the speaker from an audio file.

        Args:
            audio_file_path (str): The path to the audio file.

        Returns:
            tuple: A tuple containing the probability array and the predicted speaker.
        """
        # Perform file classification using the svmSMtemp model
        _, prob_arr, predicted_speakers = aT.file_classification(audio_file_path, model_file_path, "svm")
        
        # Convert the probability array to a list
        prob_arr = prob_arr.tolist()

        # Find the maximum probability in the array
        max_prob = max(prob_arr)

        # Find the index of the maximum probability
        index_max = prob_arr.index(max_prob)

        # Get the predicted speaker based on the index
        predicted_speaker = predicted_speakers[index_max]

        # Return the probability array and the predicted speaker
        return prob_arr, predicted_speaker


# your_instance = AccessModel(folder_path='Members2')

# your_instance.train_speaker_recognition_model()

# predicted_speaker = your_instance.get_prediction(
#         'test-samples\OMD\omar8_o.wav')


# print(f"Predicted speaker: {predicted_speaker}")

