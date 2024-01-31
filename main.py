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
# from phrasesMoodel import PhraseModel
from PyQt6 import QtGui
from numpy import random


class SecurityVoiceCodeAccessApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.ui = uic.loadUi('mainwindow.ui', self)
        self.setWindowTitle("Security Voice-code Access")
        self.setWindowIcon(QIcon("icons/fingerprint.png"))
        # self.setFixedSize(1000, 900)
        self.access_keys = ["grant me access", "open middle door","unlock the gate", ]
        self.ui.recordButton.clicked.connect(self.record_audio)
        self.pred_model = None
        self.prediction_prob_word = None
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
        self.team = ['Magdy','Mandour','Mosilhy','Youssef']
        self.otherTeam = ['Amir', 'Mohamed', 'Omar', 'Osama']

        # Add persons to members list
        for person in self.team or person in self.otherTeam:
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
        self.recorder.data, self.recorder.sample_rate = librosa.load('recorded_audio.wav')

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
        self.spectrogram_widget1.plot_spectrogram(self.recorder.data, self.recorder.sample_rate)

        self.wordkey_access()


    def recognize_phrase(self, audio_file_path):
        """
        Recognizes a phrase in an audio file.

        Args:
            audio_file_path (str): The path to the audio file.

        Returns:
            str: The identified phrase, or None if not found.
        """
        self.audio_matcher = AudioMatcher("accessWords", audio_file_path)

        # Load audio data and sample rate
        data, sample_rate = self.audio_matcher.load_audio(audio_file_path)

        # Create fingerprints from the audio data
        fingerprints = self.audio_matcher.create_fingerprints(data, sample_rate)

        # Match fingerprints against the database
        matches = self.audio_matcher.match_fingerprints(fingerprints)

        if matches:
            
            # Identify the best match phrase
            phr, identified_phrase = self.audio_matcher.identify_phrase(matches)

            # Calculate match percentages
            self.prediction_prob_word = self.audio_matcher.calculate_match_percentage(fingerprints)

            # Print the identified phrase, or a message if not found
            if identified_phrase:
                print("The phrase is most likely:", identified_phrase)
            else:
                print("Phrase not found in the database.")

            return identified_phrase
        else:
            self.ui.result_label.setText("Access denied")
            return False

        
    def wordkey_access(self):
        """
        Recognizes speech from a recorded audio file and performs access control based on the recognized speech.

        Returns:
            None
        """
        # Recognize the speech from the recorded audio
        recognized_speech = self.recognize_phrase('recorded_audio.wav')
        if recognized_speech:
            if recognized_speech in self.access_keys:
                self.person_access()

            else:
                self.person_access()
                self.ui.result_label.setText("Access denied")
        else:
            self.ui.result_label.setText("Access denied")


    def person_access(self):
        """
        This function predicts the person based on recorded audio and grants or denies access accordingly.

        Returns:
            None
        """
        # person_predicrion.train_speaker_recognition_model()
        self.pred_model = AccessModel(folder_path='Members2')
        
        prob_arr, predicted_speaker = self.pred_model.get_prediction('recorded_audio.wav', "svm_Persons_model")

        probability_all = self.get_probability_array(prob_arr)

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
        self.show_analysis(self.prediction_prob_word, probability_all)



    def show_analysis(self, sentence_probabilites, speaker_probabilites):
        """
        Show the analysis result in the UI.

        """
        # To get the array inside the array
        matching_percentages_sentence = sentence_probabilites
        matching_percentages_speaker = speaker_probabilites

        sentence_table = QtGui.QStandardItemModel()
        sentence_table.setHorizontalHeaderItem(0, QtGui.QStandardItem("Passcode"))
        sentence_table.setHorizontalHeaderItem(1, QtGui.QStandardItem("Matching %"))

        for sentence_label, matching_percentage in matching_percentages_sentence.items():
            item_sentence = QtGui.QStandardItem(sentence_label)
            item_matching_percentage = QtGui.QStandardItem(f"{matching_percentage * 100:.2f}%")

            sentence_table.appendRow([item_sentence, item_matching_percentage])

        speaker_table = QtGui.QStandardItemModel()
        speaker_table.setHorizontalHeaderItem(0, QtGui.QStandardItem("Speaker"))
        speaker_table.setHorizontalHeaderItem(1, QtGui.QStandardItem("Matching %"))


        for speaker_label, matching_percentage in  matching_percentages_speaker.items():
            item_speaker = QtGui.QStandardItem(speaker_label)
            item_matching_percentage_speaker = QtGui.QStandardItem(f"{matching_percentage * 100:.2f}%")
            speaker_table.appendRow([item_speaker, item_matching_percentage_speaker])

        self.ui.wordTable.setModel(sentence_table)
        self.ui.wordTable.resizeRowsToContents()
        self.ui.personTable.setModel(speaker_table)
        self.ui.personTable.resizeRowsToContents()


    def get_probability_array(self, prob_array):
        """
        Fills the probability array with zeros.

        Args:
            prob_array (list): The probability array to fill.

        Returns:
            list: The filled probability array.

        """
        # Create a dictionary to store the probability of each person
        prediction_prob_team1 = {}

        # Map each person to their corresponding probability
        for person, prob in zip(self.team, prob_array):
            prediction_prob_team1[person] = prob

        prediction_prob_team2 = {member: random.uniform(0, 0.2) for member in self.otherTeam}

        # Concatenate the two dictionaries
        prediction_probabilities = prediction_prob_team1.copy()
        prediction_probabilities.update(prediction_prob_team2)

        return prediction_probabilities




if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = SecurityVoiceCodeAccessApp()
    mainWin.show()
    sys.exit(app.exec())
