#!/bin/bash

DIR="$1"

mkdir -p "${DIR}/anonymized/" # Make output directory

# De-identify all existing DICOM files in directory
python3 run_dir_deidentify.py "$DIR"

echo "" # New line

# Detect new files
inotifywait -m -e create -e moved_to "$DIR" |
    while read fpath action file; do
        if [[ "$file" =~ .*dcm$ ]]; then # Is a DICOM file
            echo "Anonymizing $file"
            python3 run_deidentify.py "$fpath$file" # De-identify image
        fi
    done