import math
import os
import cv2
from ultralytics import YOLO
from people import People
from car import Car
video = os.path.join('.', 'videos', 'Casa-Ch.mp4')


video_cap = cv2.VideoCapture(video)
fps = video_cap.get(cv2.CAP_PROP_FPS)
pixels = int((24/fps)*15)

ret, frame = video_cap.read()
altura, largura, canais = frame.shape
model = YOLO("yolov8n.pt")
carro = None
persons = []
personsT = []
frameCount = 0
detection_threshold = 0.7
flag = False
centerParkX = (215 + 506) / 2
centerParkY = (89 + 380) / 2
stopedCars = []


def tracking():
    flag_2 = False
    for i in range(len(persons)):
        dist = persons[i].getdistance(bcenterX, bcenterY, frameCount, fps)
        if not flag_2 and dist < pixels:
            boxpeople = frame[y1:y2, x1:x2]
            persons[i].compare_bouding(boxpeople)
            persons[i].set_codinates(x1, x2, y1, y2)
            persons[i].set_lastframe(frameCount)
            persons[i].reverse_track()
            flag_2 = True

    if not flag_2 and len(persons) < pessoas:
        boundingboxpeople = frame[y1:y2, x1:x2]
        person1 = People(boundingboxpeople, x1, x2, y1, y2, frameCount)
        persons.append(person1)
        
    for cod in range(len(persons)):
        if persons[cod].get_tracking():
            org = (persons[cod].get_cx(), persons[cod].get_cy() - 7)
            persons[cod].reverse_track()
            cv2.circle(frame, (bcenterX, bcenterY), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(cod), org, 0, 1, (0, 0, 255), 2)


while ret:
    frameCount += 1
    ret, frame = video_cap.read()
    frame = cv2.resize(frame, (640, 480))
    results = model(frame)
    for result in results:
        pessoas = sum(1 for elemento in result.boxes.data.tolist() if elemento[-1] == 0.0)
        print("quantidade de pessoas detectadas: ", pessoas)
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            class_id = int(class_id)
            bcenterX = int((x1 + x2)/2)
            bcenterY = int((y1 + y2)/2)
            flag = math.hypot(centerParkX - (int(x1 + x2) / 2), centerParkY - (int(y1 + y2) / 2)) < 30
            for rmv in range(len(persons)):
                if persons[rmv].check_lost_track(fps, frameCount):
                    personsT.append(persons.pop(rmv))

            '''            if class_id == 2 and carro is not None and not flag:
                carro = None'''
            if class_id == 2 and carro is None and flag:
                carro = Car(frame[y1:y2, x1:x2], frameCount, bcenterX, bcenterY)
            else:
                if carro is not None:
                    if carro.getStopedTime(fps, frameCount) >= 10 and not carro.get_alerted():
                        if carro.get_alerted():
                            stopedCars.append(carro)
                        carro.viewimage(bcenterX, bcenterY)

            if class_id == 0:
                #cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 3)
                if frameCount < 1:
                    boundingBoxPeople = frame[y1:y2, x1:x2]
                    person = People(boundingBoxPeople, x1, x2, y1, y2, frameCount)
                    persons.append(person)
                else:
                    tracking()

    cv2.imshow('Camera', frame)
    cv2.waitKey(1)


video_cap.release()
cv2.destroyAllWindows()
