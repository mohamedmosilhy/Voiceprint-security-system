import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QListWidgetItem
from PyQt6.QtCore import Qt
from PyQt6 import uic
from PyQt6.QtGui import QIcon
import speech_recognition as sr
import numpy as np
import librosa
from mplwidget import MplWidget
from Audio import AudioRecorder
from matchClass import AudioMatcher
from MembersModel import AccessModel
from phrasesMoodel import PhraseModel


class SecurityVoiceCodeAccessApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.ui = uic.loadUi('mainwindow.ui', self)
        self.setWindowTitle("Security Voice-code Access")
        self.setWindowIcon(QIcon("icons/fingerprint.png"))
        # self.setFixedSize(1000, 900)
        self.access_keys = ["grant me access", "open middle door",
                            "unlock the gate", ]
        # hold the 8 persons from the combo box
        self.fingerprints = []
        self.ui.recordButton.clicked.connect(self.record_audio)
        self.ui.analysisResult.setReadOnly(True)
        self.load_ui_elements()

    def setup_widget_layout(self, widget, layout_parent):
        """
        Set up the layout for a widget within a layout parent.

        Args:
            widget: The widget to be added to the layout.
            layout_parent: The parent layout that the widget will be added to.
        """
        layout = QVBoxLayout(layout_parent)
        layout.addWidget(widget)

    def load_ui_elements(self):
        """
        Load UI elements and set up initial configurations.
        """

        # Clear result label
        self.ui.result_label.setText("")

        # Create instance of MplWidget for spectrogram
        self.spectrogram_widget1 = MplWidget()

        # Set up layout for spectrogram widget
        self.setup_widget_layout(self.spectrogram_widget1, self.ui.spectrogram)

        # Create instance of AudioRecorder for recording audio
        self.recorder = AudioRecorder(file_name='recorded_audio.wav')

        # Define list of persons
        self.persons = ['Amir', 'Magdy', 'Mandour',
                        'Mohamed', 'Mosilhy', 'Omar', 'Osama', 'Youssef']

        # Add persons to members list
        for person in self.persons:
            item = QListWidgetItem(person)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.membersList.addItem(item)

    def get_selected_persons(self):
        """
        Retrieves a list of selected persons from the members list.

        Returns:
            list: A list of strings representing the selected persons.
        """
        # Initialize an empty list to store selected items
        selected_items = []

        # Iterate through each item in the members list
        for i in range(self.membersList.count()):
            # Get the current item
            item = self.membersList.item(i)

            # Check if the item is checked
            if item.checkState() == Qt.CheckState.Checked:
                # If checked, add the item's text to the selected items list
                selected_items.append(item.text())

        # Return the list of selected items
        return selected_items

    def record_audio(self):
        """
        Records audio and processes it.
        """
        # Update the text of the recordButton to indicate that recording is in progress
        self.ui.recordButton.setText("Recording...")

        # Record audio
        self.recorder.record_audio()

        # Reset the text of the recordButton to its original state
        self.ui.recordButton.setText("Record")

        # Load the recorded audio file and its sample rate
        self.recorder.data, self.recorder.sample_rate = librosa.load(
            'recorded_audio.wav')

        # Process the audio
        self.process_audio()

    def process_audio(self):
        """
        Process the audio data.

        This function shows the spectrogram of the recorded audio, recognizes the speech,
        and then determines if the access should be granted or denied based on the match percentage
        of the recognized speech with the access keys.

        Returns:
            None
        """
        # Show spectrogram of recorded audio
        self.spectrogram_widget1.plot_spectrogram(
            self.recorder.data, self.recorder.sample_rate)

        self.word_key_access()

    def recognize_phrase(self, audio_file_path):
        """
        Recognizes a phrase in an audio file.

        Args:
            audio_file_path (str): The path to the audio file.

        Returns:
            str: The identified phrase, or None if not found.
        """

        self.prediction_prob_word = {}

        word_predicrion = PhraseModel(folder_path='accessWords')

        # person_predicrion.train_speaker_recognition_model()
        prob_arr, identified_phrase = word_predicrion.get_prediction(
            audio_file_path)

        for key, value in zip(self.access_keys, prob_arr):
            self.prediction_prob_word[key] = value

        # Print the identified phrase, or a message if not found
        if identified_phrase:
            print("The phrase is most likely:", identified_phrase)
        else:
            print("Phrase not found in the database.")

        return identified_phrase

    def word_key_access(self):
        """
        Recognizes speech from a recorded audio file and performs access control based on the recognized speech.

        Returns:
            None
        """
        # Recognize the speech from the recorded audio
        recognized_speech = self.recognize_phrase('recorded_audio.wav')

        if recognized_speech in self.access_keys:
            self.person_access()

        else:
            self.person_access()
            self.ui.result_label.setText("Access denied")

    def person_access(self):
        """
        This function predicts the person based on recorded audio and grants or denies access accordingly.

        Returns:
            None
        """
        # Example usage:
        person_predicrion = AccessModel(folder_path='Members')
        # person_predicrion.train_speaker_recognition_model()
        prob_arr, predicted_speaker = person_predicrion.get_prediction(
            'recorded_audio.wav')

        # Create a dictionary to store the probability of each person
        prediction_prob_person = {}

        # Map each person to their corresponding probability
        for person, prob in zip(self.persons, prob_arr):
            prediction_prob_person[person] = prob

        # Get the list of selected persons
        selected_persons = self.get_selected_persons()

        # Check if the predicted person is in the list of selected persons
        if selected_persons is not None and predicted_speaker in selected_persons:
            print(f"Predicted person: {predicted_speaker}")
            self.ui.result_label.setText(
                f"Hello {predicted_speaker}, Access granted")
        else:
            print("Unauthorized person detected!")
            self.ui.result_label.setText("Access denied")

        # Show the analysis of probabilities for each person
        self.show_analysis(prediction_prob_person)

    def show_analysis(self, prediction_prob_person):
        """
        Show the analysis result in the UI.

        Args:
            prediction_prob_person (dict): A dictionary containing the probabilities for each person.
        """

        # Format the probabilities
        formatted_probs = "\n".join(
            f"<b>{person}:</b> {prob:.2%} |" for person, prob in prediction_prob_person.items()
        )

        # Create a spacer line for visual separation
        spacer = "----------------------------------------------------------------------------------------------------------------------------------"

        # Format the keys and probabilities for words
        formatted_keys = "\n".join(
            f"<b>{key}:</b> {prob:.2%} |" for key, prob in self.prediction_prob_word.items()
        )

        # Concatenate the strings with spacing
        result_text = formatted_keys + f"\n{spacer}\n" + formatted_probs

        # Set the formatted text in the QLineEdit with some additional styling
        self.ui.analysisResult.setText(result_text)
        self.ui.analysisResult.setAlignment(
            Qt.AlignmentFlag.AlignTop)  # Align text to the top
        self.ui.analysisResult.setStyleSheet(
            "color: white; font-size: 16px; font-family: Arial;padding-top: 10px;line-height: 150%;")

    def recognize_speech(self, audio_filename):
        """
        Recognizes speech from an audio file using Google Web Speech API.

        Args:
            audio_filename (str): The path to the audio file.

        Returns:
            str: The recognized text, converted to lowercase for case-insensitive comparison.

        Raises:
            SpeechRecognition.UnknownValueError: If speech recognition could not understand the audio.

        """

        recognizer = sr.Recognizer()

        with sr.AudioFile(audio_filename) as source:
            audio_data = recognizer.record(source)

        try:
            recognized_text = recognizer.recognize_google(audio_data)
            return recognized_text.lower()
        except sr.UnknownValueError:
            print("Speech Recognition could not understand audio.")
            return ""

    # def calculate_word_match_percentage(self, access_keys, recognized_speech):
    #     """
    #     Calculate the word match percentage between recognized speech and a list of access keys.

    #     Args:
    #         access_keys (list): The list of access keys to compare against.
    #         recognized_speech (str): The recognized speech to compare.

    #     Returns:
    #         list: The list of match percentages between recognized speech and each access key.
    #     """
    #     match_percentages = []
    #     self.prediction_prob_word = {}  # Initialize the prediction probability dictionary

    #     for access_key in access_keys:
    #         # Split the recognized speech and access key into sets of words
    #         recognized_words = set(recognized_speech.split())
    #         access_key_words = set(access_key.split())

    #         # Find the common words between the recognized speech and access key
    #         common_words = recognized_words.intersection(access_key_words)

    #         # Calculate the match percentage using the formula:
    #         # (number of common words) / (minimum of the number of words in recognized speech and access key)
    #         match_percentage = (len(common_words) / min(len(recognized_words), len(access_key_words)))

    #         # Append the match percentage to the list
    #         match_percentages.append(match_percentage)

    #     # Assign the match percentages to the corresponding access keys in the prediction probability dictionary
    #     for key, prob in zip(self.access_keys, match_percentages):
    #         self.prediction_prob_word[key] = prob

    #     return match_percentages


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = SecurityVoiceCodeAccessApp()
    mainWin.show()
    sys.exit(app.exec())
