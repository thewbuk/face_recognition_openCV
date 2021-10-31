import cv2
import face_recognition as fr

img = cv2.imread('img/known/Bill Gates.jpg')
img2 = cv2.imread('img/known/Elon Musk.jpg')

rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = fr.face_encodings(rgb_img)[0]

rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = fr.face_encodings(rgb_img2)[0]

result = fr.compare_faces([img_encoding], img_encoding2)
print("Result: ", result)

cv2.imshow("Img", img)
cv2.imshow("Img", img2)


cv2.waitKey(0)
