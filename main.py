import cv2
import face_recognition
from simple_facerec import SimpleFacerec

cap = cv2.VideoCapture(0)

sfr = SimpleFacerec()
sfr.load_encoding_images('img/known/')

while True:
    ret, frame = cap.read()

    face_locations, face_names = sfr.detect_known_faces(frame)

    for face_loc, name in zip(face_locations, face_names):
        top, right, bottom, left = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name, (left-10, top - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
