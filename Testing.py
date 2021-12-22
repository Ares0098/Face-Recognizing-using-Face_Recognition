import face_recognition
import cv2
import os
import glob
import numpy as np
import pickle

frame_resizing = 0.25
with open("encoding/face_locker.pickle", "rb") as fopen:
    face_encoding_databases = pickle.load(fopen)

known_face_names = []
known_face_encodings = [] 
for name in face_encoding_databases:
    for encoding in face_encoding_databases[name]:
        known_face_names.append(name)
        known_face_encodings.append(encoding)

def detect_known_faces(frame, frame_resizing, known_face_encodings, known_face_names):
        small_frame = cv2.resize(frame, (0, 0), fx=frame_resizing, fy=frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / frame_resizing
        return face_locations.astype(int), face_names

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    # Detect Faces
    face_locations, face_names = detect_known_faces(frame, frame_resizing, known_face_encodings, known_face_names)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
