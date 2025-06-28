import cv2
from detection import AccidentDetectionModel
import numpy as np
import pywhatkit as pw
import winsound
import time

model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX

def startapplication():
    video = cv2.VideoCapture("test1.gif")       # 0 , test1.gif, Demo.gif , test2.webm , test4.mp4
    while True:
        ret, frame = video.read()
        
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        if(pred == "Accident"):
            prob = (round(prob[0][0]*100, 2))
            
            if(prob >= 95): 
                print ("ACCIDENT DETECTED TAKE QUICK ACTION !!")
                winsound.Beep(1500,2500)
                time.sleep(2)
                winsound.Beep(1500,2500)
                pw.sendwhatmsg_instantly("+918340777719","ACCIDENT DETECTED TAKE QUICK ACTION !!")
                winsound.Beep(1500,2500)
                time.sleep(2)
                winsound.Beep(1500,2500)
                
                break

            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, pred+" "+str(prob), (20, 30), font, 1, (255, 255, 0), 2)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            return
        cv2.imshow('Video', frame)  


if __name__ == '__main__':
    startapplication()
