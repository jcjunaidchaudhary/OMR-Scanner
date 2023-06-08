import cv2
import numpy as np
from testUtils import *

################################################################
path="img_5.4.jpg"
widthImg=550
heightImg=800
question=5
choices=4+1
# ans=[2,0,0,1,3]
ans={1:1,2:0,3:0,4:0,5:1,6:0,7:0,8:0,9:2,10:2,11:3,12:3,13:2,14:3,15:1,16:3,17:3,18:1,19:3,20:3
     ,21:1,22:3,23:0,24:0,25:2,26:2,27:0,28:0,29:1,30:3,31:2,32:0,33:1,34:3,35:0,36: 0, 37: 0, 38: 1, 39: 0, 40: 0, 
     41: 0, 42: 0, 43: 0, 44: 0, 45: 2, 46: 3, 47: 1, 48: 3, 49: 0, 50: 2, 51: 0, 52: 0, 53: 1, 54: 0, 55: 0, 56: 0, 
     57: 0, 58: 0, 59: 0, 60: 0, 61: 0, 62: 0, 63: 0, 64: 0, 65: 0, 66: 0, 67: 0, 68: 0, 69: 0, 70: 0}

# ans= {}


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
# secondContour=getCornerPoints(rectCon[2])
# print(cv2.contourArea(rectCon[0]))

count=0
for contour in rectCon:
    print(count,"--------------------------------")
    count+=1
    rectContour=getCornerPoints(contour)

    cv2.drawContours(imgFirstContour,rectContour,-1,(0,255,0),5)

    rectContour=recorder(rectContour)

    pt1=np.float32(rectContour)
    pt2=np.float32([[0,0],[400,0],[0,500],[400,500]])
    matrix=cv2.getPerspectiveTransform(pt1,pt2)
    imgWarpColored=cv2.warpPerspective(img,matrix,(390,490),1)
    imgWarpColored = imgWarpColored[15:, :]  


    #apply thresolt
    imgWarpGray=cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
    imgThresh=cv2.threshold(imgWarpGray,77,250,cv2.THRESH_BINARY_INV)[1]

    
    boxes=splitBoxes(imgThresh)
    # cv2.imshow("Test", boxes[0])
    qNum=splitBoxesForNumber(imgWarpGray)
    
    #getting nonzero values of each box
    myPixellVal=np.zeros((question,choices))
    countC=0
    countR=0
    for image in boxes:
        totalPixels=cv2.countNonZero(image)
        myPixellVal[countR][countC]=totalPixels
        countC+=1
        if countC==choices: countR+=1 ; countC=0
    
    myPixellVal = myPixellVal[:,1:] #after removing 1st column 
    print("myPixellVal",myPixellVal)

    #FINDING INDEX VALUES OF THE MARKINGS
    myAns={}
    for x in range(len(qNum)):
        arr=myPixellVal[x]
        if np.amax(arr)<50:
            myAns[qNum[x]]=6
            continue
        myIndexVal=np.where(arr==np.amax(arr))
        myAns[qNum[x]]=(myIndexVal[0][0])
    # print("myAns",myAns)
    
    # GRADING 
    grading=[]
    for x in myAns:
        if ans[x]==myAns[x]:
            grading.append(1)
        elif myAns[x]==6:
            grading.append(2)
        else: grading.append(0)
    print(grading)  

    score=(sum(grading)/question)*100 #final Grade
    print(score)

    #DISPLAYING ANSWERS
    imgResult=imgWarpColored.copy()
    imgResult=showAnswer(imgResult,myAns,grading,ans,question,choices)
    
    imRawDrawing=np.zeros_like(imgWarpColored)
    imRawDrawing=showAnswer(imRawDrawing,myAns,grading,ans,question,choices)
    # cv2.imshow("...",imRawDrawing)

    invMatrix=cv2.getPerspectiveTransform(pt2,pt1)
    imgInvWrap=cv2.warpPerspective(imRawDrawing,invMatrix,(widthImg,heightImg))


    imgFinal=cv2.addWeighted(imgFinal,1,imgInvWrap,1,0)

cv2.imshow("Image Final", imgFinal)

cv2.waitKey(000)


# if firstContour.size!=0:

#     cv2.drawContours(imgFirstContour,firstContour,-1,(0,255,0),5)
#     cv2.drawContours(imgFirstContour,secondContour,-1,(0,255,255),10)

#     firstContour=recorder(firstContour)
#     secondContour=recorder(secondContour)

#     pt1=np.float32(firstContour)
#     pt2=np.float32([[0,0],[400,0],[0,500],[400,500]])
#     matrix=cv2.getPerspectiveTransform(pt1,pt2)
#     imgWarpColored=cv2.warpPerspective(img,matrix,(390,490),1)

#     ptsecond1=np.float32(secondContour)
#     ptsecond2=np.float32([[0,0],[400,0],[0,500],[400,500]])
#     matrix=cv2.getPerspectiveTransform(ptsecond1,ptsecond2)
#     imgWarp2=cv2.warpPerspective(img,matrix,(390,490),1)

#     #apply thresolt
#     imgWarpGray=cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
#     imgThresh=cv2.threshold(imgWarpGray,80,250,cv2.THRESH_BINARY_INV)[1]
    
#     #for 2nd 
#     imgWarpGray2=cv2.cvtColor(imgWarp2,cv2.COLOR_BGR2GRAY)
#     imgThresh2=cv2.threshold(imgWarpGray2,145,260,cv2.THRESH_BINARY_INV)[1]
    
#     boxes=splitBoxes(imgThresh)
#     qNum=splitBoxesForNumber(imgWarpGray)
#     # cv2.imshow("Test", boxes[0])
    
#     #getting nonzero values of each box
#     myPixellVal=np.zeros((question,choices))
#     countC=0
#     countR=0
#     for image in boxes:
#         totalPixels=cv2.countNonZero(image)
#         myPixellVal[countR][countC]=totalPixels
#         countC+=1
#         if countC==choices: countR+=1 ; countC=0
    
#     # print(myPixellVal)
#     myPixellVal = myPixellVal[:,1:] #after removing 1st column 
#     # print("myPixellVal",myPixellVal)

#     #FINDING INDEX VALUES OF THE MARKINGS
#     myAns={}
#     # myIndex=[]
#     for x in range(len(qNum)):
#         arr=myPixellVal[x]
#         myIndexVal=np.where(arr==np.amax(arr))
#         # print(myIndexVal[0])
#         # myIndex.append(myIndexVal[0][0])
#         myAns[qNum[x]]=(myIndexVal[0][0])

#     # print("myIndex",myIndex)
#     print("myAns",myAns)


#     # # GRADING 
#     # grading=[]
#     # for x in range(question):
#     #     if ans[x]==myIndex[x]:
#     #         grading.append(1)
#     #     else: grading.append(0)
#     # print(grading)  
    
#     # GRADING 
#     grading=[]
#     for x in myAns:
#         if ans[x]==myAns[x]:
#             grading.append(1)
#         else: grading.append(0)
#     print(grading)  

#     score=(sum(grading)/question)*100 #final Grade
#     print(score)

#     #DISPLAYING ANSWERS

#     imgResult=imgWarpColored.copy()
#     imgResult=showAnswer(imgResult,myAns,grading,ans,question,choices)
#     # imgResult2=showAnswer2(imgResult)

#     cv2.imshow("Result",imgResult)
    
#     imRawDrawing=np.zeros_like(imgWarpColored)
#     imRawDrawing=showAnswer(imRawDrawing,myAns,grading,ans,question,choices)

#     # cv2.imshow("...",imRawDrawing)

#     invMatrix=cv2.getPerspectiveTransform(pt2,pt1)
#     imgInvWrap=cv2.warpPerspective(imRawDrawing,invMatrix,(widthImg,heightImg))

    
#     # imgRawGrade=np.zeros_like(imgGradeColored)
#     # cv2.putText(imgRawGrade,str(int(score))+"%",(60,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),3)
#     # invMatrixG=cv2.getPerspectiveTransform(ptG2,ptG1)
#     # imgInvGrade=cv2.warpPerspective(imgRawGrade,invMatrixG,(widthImg,heightImg))

#     imgFinal=cv2.addWeighted(imgFinal,1,imgInvWrap,1,0)
#     # imgFinal=cv2.addWeighted(imgFinal,1,imgInvGrade,1,0)




# cv2.imshow("Original", img)
# cv2.imshow("Original1", imgGray)
# cv2.imshow("Original2", imgBlur)
# cv2.imshow("Original3", imgCanny)
# cv2.imshow("Contours", imgContours)
# cv2.imshow("Contours2", imgFirstContour)
# cv2.imshow("Warp", imgWarpColored)
# cv2.imshow("Warp2", imgWarp2)
# cv2.imshow("Threshold", imgThresh)
# cv2.imshow("Threshold2", imgThresh2)
# cv2.imshow("Image Final", imgFinal)
# # cv2.imshow("raw", rectCon[0])




# cv2.waitKey(000)