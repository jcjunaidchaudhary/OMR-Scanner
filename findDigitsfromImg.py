import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd =r"C:/Program Files/Tesseract-OCR/tesseract.exe"
img1 = cv2.imread("digits.png")  # "FX2in.png"
# gry1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# (h, w) = gry1.shape[:2]
# gry1 = cv2.resize(gry1, (w*2, h*2))
# gry1 = gry1[30:(h*2), w+50:w*2]
# thr1 = cv2.threshold(gry1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
txt1 = pytesseract.image_to_string(img1, config="--psm 6 digits")
print(txt1)
# cv2.imshow("thr1", thr1)
cv2.waitKey(0)