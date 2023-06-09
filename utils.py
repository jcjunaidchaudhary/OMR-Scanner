import cv2
import numpy as np


def rectCounters(contours):
    rectCon=[]
    for i in contours:
        area =cv2.contourArea(i)
        #print("Area", area)
        if area>100:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*peri,True)
            # print("Corner Points",len(approx))
            if len(approx)==4:
                # print(True)
                rectCon.append(i)
    rectCon=sorted(rectCon,key=cv2.contourArea, reverse=True)
    # print(rectCon)
    return rectCon

def getCornerPoints(cont):
    peri=cv2.arcLength(cont,True)
    approx=cv2.approxPolyDP(cont,0.02*peri,True)
    return approx

def recorder(myPoints):
    myPoints=myPoints.reshape((4,2))
    myPointsNew=np.zeros((4,1,2),np.int32)
    add=myPoints.sum(1)
    # print(myPoints)
    # print(add)

    myPointsNew[0]=myPoints[np.argmin(add)] #[0,0]
    myPointsNew[3]=myPoints[np.argmax(add)] #[w,h]

    diff=np.diff(myPoints,axis=1)
    myPointsNew[1]=myPoints[np.argmin(diff)] #[w,0]
    myPointsNew[2]=myPoints[np.argmax(diff)] #[0,h]
    # print(myPointsNew)

    return myPointsNew
        

def splitBoxes(img):
    rows=np.vsplit(img,5)
    boxex=[]
    for r in rows:
        cols=np.hsplit(r,5)
        for box in cols:
            boxex.append(box)
    # cv2.imshow("split",rows[2])
    return boxex
    

def showAnswer(img,myIndex,grading,ans,questions,choices):
    secW=int(img.shape[1]/questions)
    secH=int(img.shape[0]/choices)

    for x in range(questions):
        myAns=myIndex[x]
        cX=(myAns*secW)+secW//2
        cY=(x*secH)+secH//2

        if grading[x]==1:
            myColor=(0,255,0)
        else:
            myColor=(0,0,255)
            correctAns=ans[x]
            dX=(correctAns*secW)+secW//2
            cv2.circle(img,(dX,cY),40,(0,255,0),cv2.FILLED)

        cv2.circle(img,(cX,cY),40,myColor,cv2.FILLED)

    return img