import cv2
import numpy as np
from utils import *

################################################################
path="test.png"
widthImg=700
heightImg=700
questions=5
choices=5
ans=[1,2,0,1,4]

################################################################

img=cv2.imread(path)

#processing
img=cv2.resize(img,(widthImg,heightImg))
imgContours=img.copy()
imgFinal=img.copy()
imgBiggestContours=img.copy()
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur=cv2.GaussianBlur(imgGray,(5,5),1)
imgCanny=cv2.Canny(imgBlur,10,30)  

#finding all contours
contours, hierarchy= cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours,contours,-1,(0,255,0),2)

#find rectangles
rectCon=rectCounters(contours)
biggestContour=getCornerPoints(rectCon[0])
# print(biggestContour.shape)
gradePoints=getCornerPoints(rectCon[1])


if biggestContour.size!=0 and gradePoints.size!=0:
    cv2.drawContours(imgBiggestContours,biggestContour,-1,(0,255,0),10)
    cv2.drawContours(imgBiggestContours,gradePoints,-1,(255,0,0),10)
    
    biggestContour=recorder(biggestContour)
    gradePoints=recorder(gradePoints)


    pt1=np.float32(biggestContour)
    pt2=np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix=cv2.getPerspectiveTransform(pt1,pt2)
    imgWarpColored=cv2.warpPerspective(img,matrix,(widthImg,heightImg))
    
    ptG1=np.float32(gradePoints)
    ptG2=np.float32([[0,0],[350,0],[0,150],[350,150]])
    matrixG=cv2.getPerspectiveTransform(ptG1,ptG2)
    imgGradeColored=cv2.warpPerspective(img,matrixG,(350,150))

    #apply thresolt
    imgWarpGray=cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
    imgThresh=cv2.threshold(imgWarpGray,155,255,cv2.THRESH_BINARY_INV)[1]

    boxex=splitBoxes(imgThresh)
    # cv2.imshow("boxex",boxex[2])

    #getting nonzero values of each box
    myPixellVal=np.zeros((questions,choices))
    countC=0
    countR=0
    for image in boxex:
        totalPixels=cv2.countNonZero(image)
        myPixellVal[countR][countC]=totalPixels
        countC+=1
        if countC==choices: countR+=1 ; countC=0
    # print(myPixellVal)

    #FINDING INDEX VALUES OF THE MARKINGS
    myIndex=[]
    for x in range(questions):
        arr=myPixellVal[x]
        myIndexVal=np.where(arr==np.amax(arr))
        # print(myIndexVal[0])
        myIndex.append(myIndexVal[0][0])
    print(myIndex)

    #GRADING 
    grading=[]
    for x in range(questions):
        if ans[x]==myIndex[x]:
            grading.append(1)
        else: grading.append(0)
    # print(grading)

    score=(sum(grading)/questions)*100 #final Grade
    # print(score)

    #DISPLAYING ANSWERS
    imgResult=imgWarpColored.copy()
    imgResult=showAnswer(imgResult,myIndex,grading,ans,questions,choices)
    
    imRawDrawing=np.zeros_like(imgWarpColored)
    imRawDrawing=showAnswer(imRawDrawing,myIndex,grading,ans,questions,choices)
    invMatrix=cv2.getPerspectiveTransform(pt2,pt1)
    imgInvWrap=cv2.warpPerspective(imRawDrawing,invMatrix,(widthImg,heightImg))
    
    imgRawGrade=np.zeros_like(imgGradeColored)
    cv2.putText(imgRawGrade,str(int(score))+"%",(60,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),3)
    invMatrixG=cv2.getPerspectiveTransform(ptG2,ptG1)
    imgInvGrade=cv2.warpPerspective(imgRawGrade,invMatrixG,(widthImg,heightImg))

    imgFinal=cv2.addWeighted(imgFinal,1,imgInvWrap,1,0)
    imgFinal=cv2.addWeighted(imgFinal,1,imgInvGrade,1,0)

# print(rectcon)
# cv2.imshow("Original", img)
# cv2.imshow("Gray", imgGray)
# cv2.imshow("Blur", imgBlur)
# cv2.imshow("Canny", imgCanny)
# cv2.imshow("Contours", imgContours)
# cv2.imshow("Biggest Con", imgBiggestContours)
# cv2.imshow("Warp", imgWarpColored)
# cv2.imshow("Grade", imgGradeColored)
# cv2.imshow("Threshold", imgThresh)
# cv2.imshow("result", imgResult)
cv2.imshow("Image Final", imgFinal)

cv2.waitKey(0)