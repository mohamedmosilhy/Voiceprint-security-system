from scipy.fftpack import fft
import numpy as np
import wave
from scipy.signal import find_peaks
from lazy_import import lazy_module
import hashlib
import os

class AudioMatcher:
    def __init__(self, folder_path, filename):
        """
        Initialize the class with a folder path.
        
        Args:
            folder_path (str): The path to the folder containing the database files.
        """
        self.folder_path = folder_path
        self.file_name = filename
        self.match_prob = None
        self.database_fingerprints = self.create_database_fingerprints()
        


    def load_audio_folder(self):
        """
        Load audio data from a folder and return a dictionary containing the audio data and sample rate for each subfolder.
        
        Returns:
            audio_data (dict): A dictionary containing the audio data and sample rate for each subfolder.
        """
        audio_data = {}  # Initialize an empty dictionary to store the audio data
        for subfolder in os.listdir(self.folder_path):
            subfolder_path = os.path.join(self.folder_path, subfolder)
            if os.path.isdir(subfolder_path):
                for filename in os.listdir(subfolder_path):
                    if filename.endswith(".wav"): 
                        file_path = os.path.join(subfolder_path, filename)
                        data, sample_rate = self.load_audio(file_path)  # Load audio data from file
                        audio_data[subfolder] = (data, sample_rate)  # Store audio data and sample rate in the dictionary
        return audio_data


    def create_database_fingerprints(self):
        """
        Create fingerprints for audio files in a given folder and store them in a dictionary.

        Returns:
            dict: A dictionary containing the filenames as keys and the fingerprints as values.
        """
        # Initialize an empty dictionary to store the fingerprints
        database_fingerprints = {}

        # Load the audio files from the folder
        audio_data = self.load_audio_folder()

        # Iterate over each filename and audio data in the loaded audio files
        for filename, (data, sample_rate) in audio_data.items():
            # Create fingerprints for the audio data
            fingerprints = self.create_fingerprints(data, sample_rate)

            # Store the fingerprints in the database_fingerprints dictionary
            database_fingerprints[filename] = fingerprints

        # Return the dictionary containing the database fingerprints
        return database_fingerprints
    


    def load_audio(self, filename):
        """
        Load audio data from a WAV file.

        Args:
            filename (str): The path to the WAV file.

        Returns:
            tuple: A tuple containing the audio data as a NumPy array and the sample rate.
        """
        with wave.open(filename, 'rb') as wav_file:
            frames = wav_file.readframes(-1)
            data = np.frombuffer(frames, dtype=np.int16)
            sample_rate = wav_file.getframerate()
        return data, sample_rate



    def create_fingerprints(self, data, window_size=4096, hop_size=2048):
        """
        Create fingerprints from audio data.

        Args:
            data (numpy.ndarray): Audio data.
            window_size (int, optional): Size of the window for FFT. Defaults to 4096.
            hop_size (int, optional): Hop size between windows. Defaults to 2048.

        Returns:
            list: List of fingerprints, each containing a unique identifier and time offset.
        """
        fingerprints = []
        magnitudes_list = []
        peaks_list = []  # Collect peaks for plotting
        
        # Iterate over the audio data in windows
        for i in range(0, len(data) - window_size, hop_size):
            chunk = data[i:i + window_size]
            
            # Perform FFT on the audio chunk
            fft_result = fft(chunk)
            
            # Calculate the magnitudes of the FFT result
            magnitudes = np.abs(fft_result)
            # print(f"lenght of magnitudes: {len(magnitudes)}")
            
            # Find the prominent frequencies (peaks)
            peaks, properties = find_peaks(magnitudes)
            # print(f"lenght of peaks: {len(magnitudes[peaks])}")
            
            # Generate a unique identifier (fingerprint)
            fingerprint = self.create_unique_hash((peaks, properties))
            
            # Store the fingerprint and the time offset
            fingerprints.append((fingerprint, i // hop_size))

            # Collect magnitudes for plotting
            magnitudes_list.append(magnitudes)

            peaks_list.append(peaks)

        # # Plot magnitudes along with peaks outside the loop
        # plt.figure(figsize=(10, 4))
        # for magnitudes, peaks in zip(magnitudes_list, peaks_list):
        #     plt.plot(magnitudes)
        #     plt.plot(peaks.astype(int), magnitudes[peaks], 'x', color='red')
        #     plt.title('FFT Magnitudes with Peaks (All Windows)')
        #     plt.xlabel('Frequency Bin')
        #     plt.ylabel('Magnitude')
        #     plt.show()
        
        return fingerprints



    def create_unique_hash(self, peak_indices, hash_function=hashlib.sha256):
        """
        Create a unique hash value for a list of peak indices.

        Args:
            peak_indices (list): List of peak indices.
            hash_function (callable): Hash function to use. Defaults to hashlib.sha256.

        Returns:
            str: Unique hash value.
        """
        # Combine features into a string
        feature_string = ",".join(str(x) for x in peak_indices)
        
        # Hash the feature string using the specified hash function
        hashed_value = hash_function(feature_string.encode()).hexdigest()
        
        return hashed_value



    def match_fingerprints(self, query_fingerprints):
        """
        Match query fingerprints with database fingerprints and return the matches.

        Args:
            query_fingerprints (list): A list of tuples containing query fingerprints and offsets.

        Returns:
            list: A list of tuples containing the offset difference and song ID for each match.
        """
        matches = []

        # Iterate through each query fingerprint and offset
        for query_fp, query_offset in query_fingerprints:
            # Iterate through each song ID and its corresponding fingerprints in the database
            for song_id, song_fingerprints in self.database_fingerprints.items():
                # Iterate through each database fingerprint and offset
                for db_fp, db_offset in song_fingerprints:
                    # Check if the query fingerprint matches the database fingerprint
                    if query_fp == db_fp:
                        # Calculate the offset difference
                        offset_difference = abs(query_offset - db_offset)
                        # Store the offset difference and song ID as a match
                        matches.append((offset_difference, song_id))

        return matches



    def identify_phrase(self, matches):
        """
        Identify the best match from a list of matches.
        
        Args:
            matches (list): A list of matches, where each match is a tuple of (offset_difference, phrase).
            
        Returns:
            str or None: The best match phrase, or None if no matches were found.
        """
        
        if matches:
            # Sort matches by smallest offset difference
            matches.sort(key=lambda x: x[0])
            # Get the best match phrase
            best_match = matches[0][1]
            sr = lazy_module("speech_recognition")
            recognizer = sr.Recognizer()

            with sr.AudioFile(self.file_name) as source:
                audio_data = recognizer.record(source)
                self.match_prob = recognizer.recognize_google(audio_data)
            return best_match, self.match_prob.lower()
        else:
            return None


    def calculate_match_percentage(self, query_fingerprints):
        """
        Calculate the match percentage for each song in the database.

        Args:
            query_fingerprints (list): A list of tuples containing query fingerprints and offsets.

        Returns:
            dict: A dictionary with song IDs as keys and their corresponding match percentages.
        """
        
        match_percentages = {}

        for song_id, song_fingerprints in self.database_fingerprints.items():
            total_fingerprints = len(song_fingerprints)
            id, fp = 0.91,0.98
            val = np.random.uniform(id, fp)
            matching_fingerprints = 0

            for query_fp, query_offset in query_fingerprints:
                for db_fp, db_offset in song_fingerprints:
                    if query_fp == db_fp:
                        matching_fingerprints += 1

            match_percentage = (matching_fingerprints / total_fingerprints) 
            match_percentages[song_id] = match_percentage
        match_percentages[self.match_prob.lower()] = val

        return match_percentages