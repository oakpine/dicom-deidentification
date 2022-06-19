import sys

from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QGridLayout, QPushButton, QLabel, QTextEdit

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from deidentify_image import Deidentify
from pydicom import dcmread

class AnonymizeGUI(QWidget):
    def __init__(self, parent = None):
        super(AnonymizeGUI, self).__init__(parent)

        self.deidentify = Deidentify()
        
        layout = QGridLayout()

        # Button to select file for anonymization
        self.btn = QPushButton("Select DICOM file")
        self.btn.clicked.connect(self.getfile)
        layout.addWidget(self.btn, 0, 2, 1, 1)

        # Label to show chosen path
        self.fname_label = QLabel()
        self.fname_label.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(self.fname_label, 0, 0, 1, 2)

        # Matplotlib figure to show image
        self.figure = Figure(figsize=(5, 3))
        self.ax = self.figure.add_subplot()
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, 1, 0, 1, 3)
        
        # Button to anonymize image
        self.btn1 = QPushButton("Anonymize Image")
        self.btn1.clicked.connect(self.deidentify_chosen)
        self.btn1.setEnabled(False)
        layout.addWidget(self.btn1, 2, 1, 1, 1)
        
        self.setLayout(layout)
        self.setWindowTitle("Anonymize DICOM Image")
        
    def getfile(self):
        # Select DICOM file
        fname = QFileDialog.getOpenFileName(self, 'Open file', 
            'c:\\',"DICOM files (*.dcm)")
        if fname[0] != '': # Catch if user closes dialog
            self.fname = fname[0]
            self.fname_label.setText(self.fname)

            # Get pixel data
            self.ds = dcmread(self.fname)
            arr = self.ds.pixel_array

            # Show DICOM image
            self.ax.cla()
            self.ax.imshow(arr, cmap="gray")
            self.canvas.draw()

            # Enable anonymize button
            self.btn1.setEnabled(True) 
            self.btn1.setText("Anonymize Image")
            self.btn1.clicked.connect(self.deidentify_chosen)

    def deidentify_chosen(self):
        # Temporarily disable button
        self.btn1.setEnabled(False)
        self.btn1.setText("Anonymizing...")

        # Make image area blank
        self.ax.cla()
        self.canvas.draw()
        self.repaint()

        # De-identify image
        self.ds = self.deidentify.deidentify_image(self.ds)
        self.ds = self.deidentify.deidentify_metadata(self.ds)

        # Get pixel data
        arr = self.ds.pixel_array

        # Show deidentified image
        self.ax.cla()
        self.ax.imshow(arr, cmap="gray")
        self.canvas.draw()

        # Turn button into a download image button
        self.btn1.setEnabled(True)
        self.btn1.setText("Download Image")
        self.btn1.clicked.disconnect()
        self.btn1.clicked.connect(self.download_image)

    def download_image(self):
        # Create default path
        out_path = self.fname.rstrip('.dcm')
        out_path += '_anonymized.dcm'

        # Save image in chosen path
        out_path = QFileDialog.getSaveFileName(self, 'Save File', out_path, "DICOM files (*.dcm)")
        if out_path[0] != '': # Catch if user closes dialog
            self.ds.save_as(out_path[0])

def main():
   app = QApplication(sys.argv)
   ex = AnonymizeGUI()
   ex.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()