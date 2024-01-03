import sounddevice as sd
import numpy as np
import wave
import pyaudio
import librosa
from joblib import load
from scipy import stats

class AudioRecorder:
    def __init__(self,file_name , duration=5, sample_rate=44100):
        self.duration = duration
        self.sample_rate = sample_rate
        self.file_name = file_name
        self.data = None



    def record_audio(self, duration=4, channels=1, sample_rate=44100, chunk_size=1024):
        """
        Records audio from the default input device for the specified duration.

        Args:
            duration (float): The duration of the recording in seconds (default is 4).
            channels (int): The number of audio channels (default is 1).
            sample_rate (int): The sample rate of the audio (default is 44100).
            chunk_size (int): The number of frames per buffer (default is 1024).
        """

        # Initialize PyAudio
        p = pyaudio.PyAudio()

        # Open stream
        stream = p.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

        print("Recording...")

        # Buffer for storing audio frames
        frames = []

        # Record audio frames
        for i in range(0, int(sample_rate / chunk_size * duration)):
            data = stream.read(chunk_size)
            frames.append(data)

        print("Recording done.")

        # Stop stream
        stream.stop_stream()
        stream.close()

        # Terminate PyAudio
        p.terminate()

        # Save the recorded audio as a WAV file
        with wave.open(self.file_name, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))
        

    def get_audio_data(self):
        # Load the audio file
        data, sr = librosa.load(self.file_name, sr=self.sample_rate)
        return data , sr


# #######################################################################################################
# #                                 Load the trained model and scaler                                   #
# #######################################################################################################
    
#     @staticmethod
#     def load_model(model_path):
#         """
#         Load the trained model and scaler from the given model path.

#         Args:
#             model_path (str): The path to the trained model.

#         Returns:
#             Tuple: A tuple containing the loaded model and scaler.
#         """
#         # Load the trained model and scaler using joblib
#         loaded_model, loaded_scaler = load(model_path)
#         return loaded_model, loaded_scaler



#     def preprocess_audio(self, audio_path, target_sr=44100, target_duration=5):
#         """
#         Preprocesses audio file by loading, resampling, and trimming or padding it.

#         Args:
#             audio_path (str): Path to the audio file.
#             target_sr (int, optional): Target sample rate. Defaults to 44100.
#             target_duration (int, optional): Target duration in seconds. Defaults to 5.

#         Returns:
#             tuple: Preprocessed audio data and target sample rate.
#         """
#         # Load the audio data and sample rate
#         data, sample_rate = librosa.load(audio_path)

#         # Resample if necessary
#         if sample_rate != target_sr:
#             data = librosa.resample(data, orig_sr=sample_rate, target_sr=target_sr)

#         # Trim or pad the audio data to the target duration
#         if len(data) > target_duration * target_sr:
#             data = data[:target_duration * target_sr]
#         else:
#             data = np.pad(data, (0, target_duration * target_sr - len(data)), 'constant')

#         return data, target_sr



#     def predict(self, audio_path):
#         """
#         Predict the class label and probability for a given audio file.

#         Args:
#             audio_path (str): Path to the audio file.

#         Returns:
#             tuple: A tuple containing the predicted class label and probability.
#         """

#         # Preprocess the audio
#         data, sample_rate =  self.preprocess_audio(audio_path)

#         # Extract features
#         features = np.array([
#                     np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate)),
#                     np.mean(librosa.feature.chroma_cqt(y=data, sr=sample_rate)),
#                     np.mean(librosa.feature.delta(librosa.feature.mfcc(y=data), order=1)),
#                     np.mean(librosa.effects.harmonic(y=data)),
#                 ]).reshape(1, -1)

#         # Scale features using the loaded scaler
#         scaled_features = self.loaded_scaler.transform(features)

#         # Make a prediction using the loaded model
#         prediction = self.loaded_model.predict(scaled_features)
#         prediction_prob = self.loaded_model.predict_proba(scaled_features)[0]

#         return prediction[0], prediction_prob
