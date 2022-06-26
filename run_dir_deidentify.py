import sys
import glob
import os

from pydicom import dcmread
from deidentify_image import Deidentify

path = sys.argv[1] # Directory

# Find all DICOM files in directory
files = glob.glob(path + "/*.dcm")

# Loop through each file in directory
for file in files:
    fname = os.path.basename(file) # Get file name
    dir = os.path.dirname(file) # Get directory name

    print("Anonymizing " + fname)
    ds = dcmread(file)

    # Deidentify image
    deidentify = Deidentify()
    ds = deidentify.deidentify_image(ds)
    ds = deidentify.deidentify_metadata(ds)

    # Save de-identified version
    ds.save_as(os.path.join(dir, "anonymized", fname))