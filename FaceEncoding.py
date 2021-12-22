import face_recognition
import cv2
import os
import glob
import numpy as np
import pickle

path = "images2/"
known_face_encodings = []
known_face_names = []
face_locker = {}

def load_encoding_images(images_path):
        # Load Images
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # Store image encoding
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get encoding
            img_encoding = face_recognition.face_encodings(rgb_img)

            # Store file name and file encoding
            known_face_encodings.append(img_encoding)
        print("Encoding images loaded")

        return known_face_encodings

def load_encoding_name(images_path) :
    # Load Images
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # Store image names
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)

            # Store file name and file encoding
            known_face_names.append(filename)
        print("Encoding names loaded")

        return known_face_names


file_face_encodings = load_encoding_images(path)
print("========================================")
file_name_encodings = load_encoding_name(path)
print("========================================\n")

for i in range(len(file_face_encodings)) :
    face_locker[file_name_encodings[i]] = file_face_encodings[i]

with open("encoding/face_locker.pickle", "wb") as fsave:
    pickle.dump(face_locker,fsave)

print("------------------------------------------")
print("                File Saved                ")
print("------------------------------------------")
