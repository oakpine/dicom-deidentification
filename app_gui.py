import sys
import glob
import os

from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QGridLayout, QVBoxLayout, QPushButton, QLabel, \
                            QTabWidget, QListWidget, QListWidgetItem
from PyQt5 import QtCore
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from deidentify_image import Deidentify
from pydicom import dcmread

class AnonymizeGUI(QWidget):
    def __init__(self, parent = None):
        super(AnonymizeGUI, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        self.deidentify = Deidentify()

        # Initialize tabs
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()

        # Add tabs
        self.tabs.addTab(self.tab1,"Image")
        self.tabs.addTab(self.tab2,"Directory")
        
        self.tab1.layout = QGridLayout()
        self.tab2.layout = QGridLayout()

        ##### Tab 1 #####

        # Button to select file for anonymization
        self.select_btn1 = QPushButton("Select DICOM file")
        self.select_btn1.clicked.connect(self.get_file)
        self.tab1.layout.addWidget(self.select_btn1, 0, 2, 1, 1)

        # Label to show chosen path
        self.fname_label = QLabel()
        self.fname_label.setStyleSheet("border: 1px solid gray;")
        self.tab1.layout.addWidget(self.fname_label, 0, 0, 1, 2)

        # Matplotlib figure to show image
        self.figure = Figure(figsize=(5, 3))
        self.ax = self.figure.add_subplot()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.tab1.layout.addWidget(self.canvas, 1, 0, 1, 3)

        # Button to anonymize image
        self.anonymize_btn1 = QPushButton("Anonymize Image")
        self.anonymize_btn1.clicked.connect(self.deidentify_file)
        self.anonymize_btn1.setEnabled(False)
        self.tab1.layout.addWidget(self.anonymize_btn1, 2, 1, 1, 1)

        ##### Tab 2 #####

        # Button to select directory for anonymization
        self.select_btn2 = QPushButton("Select directory")
        self.select_btn2.clicked.connect(self.get_dir)
        self.tab2.layout.addWidget(self.select_btn2, 0, 2, 1, 1)

        # Label to show chosen path
        self.dir_label = QLabel()
        self.dir_label.setStyleSheet("border: 1px solid gray;")
        self.tab2.layout.addWidget(self.dir_label, 0, 0, 1, 2)

        # Label to show files in directory
        self.file_list = QListWidget()
        self.tab2.layout.addWidget(self.file_list, 1, 0, 1, 3)

        # Button to anonymize images
        self.anonymize_btn2 = QPushButton("Anonymize Images")
        self.anonymize_btn2.clicked.connect(self.deidentify_dir)
        self.anonymize_btn2.setEnabled(False)
        self.tab2.layout.addWidget(self.anonymize_btn2, 2, 1, 1, 1)

        self.tab1.setLayout(self.tab1.layout)
        self.tab2.setLayout(self.tab2.layout)
        
        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        self.setWindowTitle("Anonymize DICOM Image")
        
    def get_file(self):
        # Select DICOM file
        fname = QFileDialog.getOpenFileName(self, 'Open File', 
            "/home","DICOM files (*.dcm)")
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
            self.anonymize_btn1.setEnabled(True) 
            self.anonymize_btn1.setText("Anonymize Image")
            self.anonymize_btn1.clicked.connect(self.deidentify_file)

    def get_dir(self):
        # Select directory
        dir_name = QFileDialog.getExistingDirectory(self, "Open Directory",
                                       "/home",
                                       QFileDialog.ShowDirsOnly
                                       | QFileDialog.DontResolveSymlinks)
        if dir_name != '': # Catch if user closes dialog
            self.dir_name = dir_name
            self.dir_label.setText(self.dir_name)

            self.dir_fnames = glob.glob(self.dir_name + "/*.dcm")

            self.file_list.clear() # Clear QListWidget
            self.anonymize_btn2.setText("Anonymize Images")

            if len(self.dir_fnames) == 0: # No DICOM files in the folder
                QListWidgetItem("No DICOM files found in the chosen directory.", self.file_list)
                self.anonymize_btn2.setEnabled(False) # Disable anonymize button 
            else:
                # Display all file names in a separate line
                for f in self.dir_fnames:
                    f = os.path.basename(f) # Get file name
                    item = QListWidgetItem(f, self.file_list)
                    item.setFlags(QtCore.Qt.NoItemFlags) # Disable interaction with list items
                    item.setForeground(Qt.black)

                # Enable anonymize button
                self.anonymize_btn2.setEnabled(True) 

    def deidentify_file(self):
        # Temporarily disable button
        self.anonymize_btn1.setEnabled(False)
        self.anonymize_btn1.setText("Anonymizing...")

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
        self.anonymize_btn1.setEnabled(True)
        self.anonymize_btn1.setText("Download Image")
        self.anonymize_btn1.clicked.disconnect()
        self.anonymize_btn1.clicked.connect(self.download_image)

    def deidentify_dir(self):
        # Choose directory to save images
        save_dir = QFileDialog.getExistingDirectory(self, "Choose directory to save images",
                                       self.dir_name,
                                       QFileDialog.ShowDirsOnly
                                       | QFileDialog.DontResolveSymlinks)
        if save_dir != '': # Catch if user closes dialog
            # Temporarily disable button
            self.anonymize_btn2.setEnabled(False)
            self.anonymize_btn2.setText("Anonymizing...")
            self.repaint()

            # Loop through DICOM files in directory
            for i in range(len(self.dir_fnames)):
                file = self.dir_fnames[i]

                # Read image
                ds = dcmread(file)

                # De-identify image
                ds = self.deidentify.deidentify_image(ds)
                ds = self.deidentify.deidentify_metadata(ds)

                # Save image
                file_name = os.path.basename(file) # Get file name
                ds.save_as(save_dir + '/' + file_name)

                # Show that file has been anonymized
                list_item = self.file_list.item(i)
                list_item.setText("Anonymized --- " + list_item.text())
                self.repaint()

            self.anonymize_btn2.setText("Done")

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