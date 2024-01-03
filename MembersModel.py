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
        speakers = list(self.audio_data.keys())
        paths = [os.path.join(self.folder_path, speaker) for speaker in speakers]

        aT.extract_features_and_train(paths, 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False)

    
    def get_prediction(self, audio_file_path):
        _ , prob_arr, predicted_speakers = aT.file_classification(audio_file_path, "svmSMtemp", "svm")
        prob_arr = prob_arr.tolist()

        max_prob = max(prob_arr)

        index_max = prob_arr.index(max_prob)

        predicted_speaker = predicted_speakers[index_max]
        return prob_arr , predicted_speaker


# your_instance = AccessModel(folder_path='Members')

# your_instance.train_speaker_recognition_model()

# predicted_speaker = your_instance.get_prediction(
#         'test-samples\OMD\omar8_o.wav')


# print(f"Predicted speaker: {predicted_speaker}")

