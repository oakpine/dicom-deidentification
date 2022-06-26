import sys
import os
from pydicom import dcmread
from deidentify_image import Deidentify

path = sys.argv[1] # Path to file

ds = dcmread(path)

# Deidentify image
deidentify = Deidentify()
ds = deidentify.deidentify_image(ds)
ds = deidentify.deidentify_metadata(ds)

file = os.path.basename(path) # Get file name
dir = os.path.dirname(path) # Get directory name

# Save de-identified version
ds.save_as(os.path.join(dir, "anonymized", file))