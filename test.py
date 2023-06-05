import cv2
import numpy as np
from testUtils import *

################################################################
path="img_5.2.jpg"
widthImg=550
heightImg=800
question=5
choices=4+1
ans=[2,0,0,1,3]


################################################################

img=cv2.imread(path)

#processing
img=cv2.resize(img,(widthImg,heightImg))
imgContours=img.copy()
imgFinal=img.copy()
imgFirstContour=img.copy()
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur=cv2.GaussianBlur(imgGray,(5,5),1)
imgCanny=cv2.Canny(imgBlur,10,50)  

contours, hierarchy= cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours,contours,-1,(0,255,0),6)

rectCon=rectCounters(contours)
firstContour=getCornerPoints(rectCon[0])
secondContour=getCornerPoints(rectCon[1])
# thirdContour=getCornerPoints(rectCon[1])



if firstContour.size!=0:

    cv2.drawContours(imgFirstContour,firstContour,-1,(0,255,0),5)
    cv2.drawContours(imgFirstContour,secondContour,-1,(0,255,255),10)

    firstContour=recorder(firstContour)
    secondContour=recorder(secondContour)

    pt1=np.float32(firstContour)
    pt2=np.float32([[0,0],[400,0],[0,500],[400,500]])
    matrix=cv2.getPerspectiveTransform(pt1,pt2)
    imgWarpColored=cv2.warpPerspective(img,matrix,(390,490),1)

    ptsecond1=np.float32(secondContour)
    ptsecond2=np.float32([[0,0],[400,0],[0,500],[400,500]])
    matrix=cv2.getPerspectiveTransform(ptsecond1,ptsecond2)
    imgWarp2=cv2.warpPerspective(img,matrix,(390,490),1)

    #apply thresolt
    imgWarpGray=cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
    imgThresh=cv2.threshold(imgWarpGray,80,250,cv2.THRESH_BINARY_INV)[1]
    
    boxes=splitBoxes(imgThresh)
    # cv2.imshow("Test", boxes[0])
    
    #getting nonzero values of each box
    myPixellVal=np.zeros((question,choices))
    countC=0
    countR=0
    for image in boxes:
        totalPixels=cv2.countNonZero(image)
        myPixellVal[countR][countC]=totalPixels
        countC+=1
        if countC==choices: countR+=1 ; countC=0

    print(myPixellVal)
    myPixellVal = myPixellVal[:,1:]
    # print(myPixellVal)

    #FINDING INDEX VALUES OF THE MARKINGS
    myIndex=[]
    for x in range(question):
        arr=myPixellVal[x]
        myIndexVal=np.where(arr==np.amax(arr))
        # print(myIndexVal[0])
        myIndex.append(myIndexVal[0][0])
    print(myIndex)


    #GRADING 
    grading=[]
    for x in range(question):
        if ans[x]==myIndex[x]:
            grading.append(1)
        else: grading.append(0)
    print(grading)  

    score=(sum(grading)/question)*100 #final Grade
    print(score)

    #DISPLAYING ANSWERS

    imgResult=imgWarpColored.copy()
    imgResult=showAnswer(imgResult,myIndex,grading,ans,question,choices)
    
    imRawDrawing=np.zeros_like(imgWarpColored)
    imRawDrawing=showAnswer(imRawDrawing,myIndex,grading,ans,question,choices)


    invMatrix=cv2.getPerspectiveTransform(pt2,pt1)
    imgInvWrap=cv2.warpPerspective(imRawDrawing,invMatrix,(widthImg,heightImg))
    
    # imgRawGrade=np.zeros_like(imgGradeColored)
    # cv2.putText(imgRawGrade,str(int(score))+"%",(60,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),3)
    # invMatrixG=cv2.getPerspectiveTransform(ptG2,ptG1)
    # imgInvGrade=cv2.warpPerspective(imgRawGrade,invMatrixG,(widthImg,heightImg))

    imgFinal=cv2.addWeighted(imgFinal,1,imgInvWrap,1,0)
    # imgFinal=cv2.addWeighted(imgFinal,1,imgInvGrade,1,0)




# cv2.imshow("Original", img)
# cv2.imshow("Original1", imgGray)
# cv2.imshow("Original2", imgBlur)
# cv2.imshow("Original3", imgCanny)
# cv2.imshow("Contours", imgContours)
# cv2.imshow("Contours2", imgFirstContour)
cv2.imshow("Warp", imgWarpColored)
# cv2.imshow("Warp2", imgWarp2)
# cv2.imshow("Threshold", imgThresh)
# cv2.imshow("Threshold2", imgInvWrap)
cv2.imshow("Image Final", imgFinal)
# cv2.imshow("raw", imRawDrawing)




cv2.waitKey(0)