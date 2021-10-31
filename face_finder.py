import cv2
import face_recognition
from simple_facerec import SimpleFacerec

cap = cv2.VideoCapture(0)

sfr = SimpleFacerec()

while True:
    ret, frame = cap.read()

    face_locations, face_names = sfr.detect_known_faces(frame)

    for face_loc, name in zip(face_locations, face_names):
        top, left, bottom, right = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
