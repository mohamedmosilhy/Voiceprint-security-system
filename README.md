# Security Voice-code Access

This project implements a Security Voice-code Access system with two operation modes using fingerprint and spectrogram concepts. The software is designed to be trained on 8 individuals and can operate in two modes: Security Voice Code and Security Voice Fingerprint.

## Features

### 1. Security Voice Code (Mode 1)

- The system allows access only when a specific pass-code sentence is spoken.
- Valid passcode sentences:
  - "Open middle door"
  - "Unlock the gate"
  - "Grant me access"
- Custom passcodes can be set, ensuring no similar words among the chosen sentences.

### 2. Security Voice Fingerprint (Mode 2)

- Access is granted based on the voice fingerprint of specific individual(s) who say the valid passcode sentence.
- Users can select which individuals among the original 8 are granted access.

### 3. User Interface

- Button to start recording the voice-code.
- Spectrogram viewer for visualizing the spoken voice-code in real-time.
- Summary for analysis results, including:
  - A table with match percentages for each of the three passcode sentences.
  - A table with match percentages for each of the 8 saved individuals.
- An indicator for the algorithm results: "Access gained" or "Access denied."

## Implementation Details

### Software Components

1. **Voice Recording Module:**
   - Captures audio input.
   - Provides start and stop recording functionality.

2. **Speech-to-Text Module:**
   - Converts recorded voice into text for analysis.
   - Checks for a match with predefined passcode sentences.

3. **Spectrogram Viewer:**
   - Utilizes `MplWidget` for visualizing the spectrogram of the recorded voice.
   - Displays the spectrogram in real-time during voice recording.

4. **Voice Fingerprinting Module:**
   - Extracts voice fingerprints from the recorded voice.
   - Compares the voiceprint with stored voiceprints of registered individuals.

5. **Results Display Module:**
   - Generates and displays tables showing matching scores.
   - Clearly indicates whether access is granted or denied based on analysis results.

6. **Settings UI:**
   - Allows the user to choose the operation mode (Security Voice Code or Security Voice Fingerprint).
   - In Security Voice Fingerprint mode, enables the user to select one or more individuals for access.

### Technologies Used

- **Programming Language:** Python
- **UI Framework:** PyQt6
- **Signal Processing Libraries:** NumPy, SciPy, Matplotlib, librosa
- **Speech Recognition Library:** SpeechRecognition

## Usage

1. Run the application using Python: `python main.py`.
2. Use the provided UI to start recording voice-code.
3. View the real-time spectrogram and analysis results.
4. Access will be granted or denied based on the chosen mode and recognition results.

## How to Run

1. Install the required dependencies using 
```bash
pip install -r requirements.txt
```
2. Run the application using 
```bash
python main.py
```
