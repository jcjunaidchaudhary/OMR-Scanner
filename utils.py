import cv2
import numpy as np


def rectCounters(contours):
    rectCon=[]
    for i in contours:
        area =cv2.contourArea(i)
        #print("Area", area)
        if area>50:
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
        