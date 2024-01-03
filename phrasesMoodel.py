import os
import numpy as np
from pyAudioAnalysis import audioTrainTest as aT
import wave

class PhraseModel:
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

        # Extract features from audio data and train speaker recognition model
        aT.extract_features_and_train(
            paths, 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmout", False)

    
    def get_prediction(self, audio_file_path):
        """
        Get the prediction for the speaker in an audio file.

        Parameters:
        - audio_file_path (str): The path to the audio file.

        Returns:
        - prob_arr (list): The probability array for each speaker.
        - predicted_speaker (str): The predicted speaker.
        """

        # Get the classification results from the audio file
        _, prob_arr, predicted_speakers = aT.file_classification(audio_file_path, "svmout", "svm")

        # Convert the probability array to a list
        prob_arr = prob_arr.tolist()

        # Find the maximum probability and its index in the array
        max_prob = max(prob_arr)
        index_max = prob_arr.index(max_prob)

        # Get the predicted speaker based on the maximum probability
        predicted_speaker = predicted_speakers[index_max]

        # Return the probability array and the predicted speaker
        return prob_arr, predicted_speaker

        
# your_instance = AccessModel(folder_path='accessWords')
# your_instance.train_speaker_recognition_model()
# predicted_speaker = your_instance.get_prediction('test-samples\OMD\omar8_o.wav')


# print(f"Predicted speaker: {predicted_speaker}")