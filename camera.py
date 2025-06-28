import cv2
from detection import AccidentDetectionModel
import numpy as np

model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX

def startapplication():
    
    video = cv2.VideoCapture('Demo.gif')      #( 0 , 'test2.webm' , 'test4.mp4' , 'Demo.gif' )
    
    while True:
        ret, frame = video.read()
        
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        if(pred == "Accident"):
            prob = (round(prob[0][0]*100, 2))
            
            if(prob >= 90): 
                print ("Accident Detected")

            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, pred+" "+str(prob), (20, 30), font, 1, (255, 255, 0), 2)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            return
        cv2.imshow('Video', frame)  


if __name__ == '__main__':
    startapplication()