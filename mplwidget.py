from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import librosa.display



class MplWidget(QWidget):
    def __init__(self, parent=None):
        """
        Initialize the class.

        Args:
            parent (QWidget): The parent widget. Defaults to None.
        """
        super().__init__(parent)

        # Create a canvas for plotting
        self.canvas = FigureCanvas(Figure(facecolor='none'))
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                QtWidgets.QSizePolicy.Policy.Expanding)

        # Create a vertical layout and add the canvas to it
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)

        # Create and add axes to the canvas
        self.canvas.axes = self.canvas.figure.add_subplot(111)

        # Set the color of the axes to white
        self.canvas.axes.tick_params(axis='both', colors='white')

        # Set the layout of the widget
        self.setLayout(vertical_layout)

        # Initialize the colorbar to None
        self.colorbar = None



    def plot_spectrogram(self, audio_data, sample_rate, x_label='Time', y_label='Frequency'):
        """
        Plot the spectrogram of the given audio data.

        Args:
            audio_data (numpy.ndarray): The audio data.
            sample_rate (int): The sample rate of the audio data.
            x_label (str, optional): The label for the x-axis. Defaults to 'Time'.
            y_label (str, optional): The label for the y-axis. Defaults to 'Frequency'.
        """
        # Clear the axes
        self.canvas.axes.clear()

        # Compute the spectrogram
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)

        # Plot the spectrogram
        pcm = librosa.display.specshow(spectrogram, sr=sample_rate, x_axis='time', y_axis='log', ax=self.canvas.axes)
        self.canvas.axes.set_xlabel(x_label, color='white')
        self.canvas.axes.set_ylabel(y_label, color='white')

        if not self.colorbar:
            # Add new colorbar
            self.colorbar = self.canvas.figure.colorbar(pcm, ax=self.canvas.axes)
            self.colorbar.set_label('Intensity (dB)', color='white')
            self.colorbar.ax.yaxis.set_tick_params(color='white')
            self.colorbar.ax.tick_params(axis='y', colors='white')
            self.colorbar.outline.set_edgecolor('white')

        self.canvas.draw()

    def clear(self):
        self.canvas.axes.clear()

        self.canvas.draw()


