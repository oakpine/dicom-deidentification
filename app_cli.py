from pydicom import dcmread
from deidentify_image import Deidentify

# path = 'img/ID_0006.dcm'
# path = 'img/nema_img11.dcm'
# path = 'img/CT-MONO2-8-abdo'
# path = 'img/CT-MONO2-16-ankle'
path = input("Path to DICOM file to anonymize: ")

# Ask for file path until it is valid
while True:
    try:
        ds = dcmread(path)
        break
    except FileNotFoundError:
        print("\nThere is no DICOM file at the specified path.")
        path = input("Path to DICOM file to anonymize: ")

deidentify = Deidentify()
ds = deidentify.deidentify_image(ds)
ds = deidentify.deidentify_metadata(ds)

# Save de-identified image
out_path = path.rstrip('.dcm')
out_path += '_anonymized.dcm'
ds.save_as(out_path)

print("Anonymized file saved at '" + out_path + "'")