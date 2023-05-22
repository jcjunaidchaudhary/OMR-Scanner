import cv2
import numpy as np
from utils import *

################################################################
path="sheet4.jpg"
widthImg=700
heightImg=700
################################################################

img=cv2.imread(path)

#processing
img=cv2.resize(img,(widthImg,heightImg))
imgContours=img.copy()
imgBiggestContours=img.copy()
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur=cv2.GaussianBlur(imgGray,(5,5),1)
imgCanny=cv2.Canny(imgBlur,10,50)  

#finding all contours
contours, hierarchy= cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours,contours,-1,(0,255,0),10)

#find rectangles
rectCon=rectCounters(contours)
biggestContour=getCornerPoints(rectCon[0])
# print(biggestContour.shape)
gradePoints=getCornerPoints(rectCon[1])
# gradePoints2=getCornerPoints(rectCon[2])
# gradePoints3=getCornerPoints(rectCon[3])
# gradePoints4=getCornerPoints(rectCon[4])
# gradePoints5=getCornerPoints(rectCon[5])


if biggestContour.size!=0 and gradePoints.size!=0:
    cv2.drawContours(imgBiggestContours,biggestContour,-1,(0,255,0),10)
    cv2.drawContours(imgBiggestContours,gradePoints,-1,(255,0,0),10)
    # cv2.drawContours(imgBiggestContours,gradePoints2,-1,(255,0,0),10)
    # cv2.drawContours(imgBiggestContours,gradePoints3,-1,(255,0,0),10)
    # cv2.drawContours(imgBiggestContours,gradePoints4,-1,(255,0,0),10)
    # cv2.drawContours(imgBiggestContours,gradePoints5,-1,(255,0,0),10)
    
    biggestContour=recorder(biggestContour)
    gradePoints=recorder(gradePoints)
    # gradePoints2=recorder(gradePoints2)
    # gradePoints3=recorder(gradePoints3)
    # gradePoints4=recorder(gradePoints4)
    # gradePoints5=recorder(gradePoints5)

    pt1=np.float32(biggestContour)
    pt2=np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix=cv2.getPerspectiveTransform(pt1,pt2)
    imgWarpColored=cv2.warpPerspective(img,matrix,(widthImg,heightImg))
    
    ptG1=np.float32(gradePoints)
    ptG2=np.float32([[0,0],[200,0],[0,heightImg],[200,heightImg]])
    matrixG=cv2.getPerspectiveTransform(ptG1,ptG2)
    imgGradeColored=cv2.warpPerspective(img,matrixG,(200,heightImg))

# print(rectcon)
# cv2.imshow("Original", img)
# cv2.imshow("Original2", imgGray)
# cv2.imshow("Original3", imgBlur)
# cv2.imshow("Original4", imgCanny)
# cv2.imshow("Original5", imgContours)
# cv2.imshow("Original6", imgBiggestContours)
cv2.imshow("Original7", imgWarpColored)
cv2.imshow("Original8", imgGradeColored)

cv2.waitKey(0)