import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd =r"C:/Program Files/Tesseract-OCR/tesseract.exe"

def rectCounters(contours):
    rectCon=[]
    for i in contours:
        area =cv2.contourArea(i)
       
        if area>10000:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*peri,True)
            # print("Corner Points",len(approx))
            if len(approx)==4:
                # print(True)
                rectCon.append(i)
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
    boxes=[]
    # innter_rows=np.vsplit(rows,5)
    # cv2.imshow("split",rows[0])
    # cv2.imshow("split2",innter_rows[1])
    for r in rows:
        cols=np.hsplit(r,5)
        for box in cols:
            boxes.append(box)
        # cv2.imshow("split",box)
    return boxes

def splitBoxesForNumber(img):
    rows=np.vsplit(img,5)
    qNumber=[]
    # cv2.imshow("split",rows[0])
    count=0
    for r in rows:
        cols=np.hsplit(r,5)
        resized_image = cv2.resize(cols[0], None, fx=2, fy=1)
        
        
        # print("Shape of the image", resized_image.shape)
        resized_image = resized_image[:, 9:]  
        
        # cv2.imshow(f"splitNumber{count}",resized_image)
        
        txt1 = pytesseract.image_to_string(resized_image, config="--psm 6 digits")
        print(txt1)
        
        try:
            qNumber.append(int(txt1))
        except:
            pass
        
        count+=1
    
    print(qNumber)

    qNumber=[i for i in range(qNumber[-1]-4,qNumber[-1]+1)]
    
    return qNumber

# def showAnswer(img,myIndex,grading,ans,questions,choices):
#     print("Question",questions)
#     print("Choices",choices)
#     secW=int(img.shape[1]/questions)
#     secH=int(img.shape[0]/choices)

#     for x in range(questions):
#         myAns=myIndex[x]+1
#         print(myAns,"Answer")
#         cX=(myAns*secW)+secW//2
#         cY=(x*secH)+secH//2

#         if grading[x]==1:
#             myColor=(0,255,0)
#         else:
#             myColor=(0,0,255)
#             correctAns=ans[x]+1
#             print ("correct",correctAns)
#             dX=(correctAns*secW)+secW//2
#             cv2.circle(img,(dX,cY),15,(0,255,0),cv2.FILLED)

#         cv2.circle(img,(cX,cY),15,myColor,cv2.FILLED)

#     return img

def showAnswer(img,myAnswer,grading,ans,questions,choices):
    questions=5

    secW=int(img.shape[1]/questions)
    secH=int(img.shape[0]/choices)

    indx=0
    for x in myAnswer:
        myAns=myAnswer[x]+1
        cX=(myAns*secW)+secW//2
        cY=(indx*secH)+secH//2

        if grading[indx]==1:
            myColor=(0,255,0)
        else:
            myColor=(0,0,255)
            correctAns=ans[x]+1
            dX=(correctAns*secW)+secW//2
            cv2.circle(img,(dX,cY),15,(0,255,0),cv2.FILLED)

        cv2.circle(img,(cX,cY),15,myColor,cv2.FILLED)
        indx+=1
    return img


