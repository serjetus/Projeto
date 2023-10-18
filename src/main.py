import os
import random

import cv2
from ultralytics import YOLO
from people import People

video = os.path.join('.', 'videos', 'Casa-Ch.mp4')


video_cap = cv2.VideoCapture(video)
fps = video_cap.get(cv2.CAP_PROP_FPS)
pixels = int((24/fps)*15)

ret, frame = video_cap.read()

model = YOLO("yolov8n.pt")

persons = []
frameCount = 0
detection_threshold = 0.7


while ret:
    frameCount += 1
    ret, frame = video_cap.read()
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
            largura = 2
            altura = 2
            z1 = bcenterX - largura // 2
            u1 = bcenterY - altura // 2
            z2 = z1 + largura
            u2 = u1 + altura
            center = frame[u1:u2, z1:z2]

            if class_id == 0:
                #cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 3)
                if frameCount < 1:
                    boundingBoxPeople = frame[y1:y2, x1:x2]
                    person = People(boundingBoxPeople, x1, x2, y1, y2, frameCount, center)
                    persons.append(person)

                else:
                    flag = False
                    for i in range(len(persons)):
                        comp = persons[i].compare_centerpx(center)
                        print(comp)
                        if not flag and persons[i].getdistance(bcenterX, bcenterY, frameCount, fps) < pixels:
                            print("a pessoa", i, " se moveu")
                            persons[i].set_codinates(x1, x2, y1, y2)
                            persons[i].set_lastframe(frameCount)
                            persons[i].reverse_track()
                            flag = True
                    if not flag and len(persons) < pessoas:
                        boundingBoxPeople = frame[y1:y2, x1:x2]
                        person = People(boundingBoxPeople, x1, x2, y1, y2, frameCount, center)
                        persons.append(person)

            for cod in range(len(persons)):
                if persons[cod].get_tracking():
                    org = (persons[cod].get_cx(), persons[cod].get_cy() - 7)
                    persons[cod].reverse_track()
                    cv2.circle(frame, (bcenterX, bcenterY), 5, (0, 255, 0), -1)
                    cv2.putText(frame, str(cod), org, 0, 1, (0, 0, 255), 2)

    cv2.imshow('teste', frame)
    cv2.waitKey(1)


video_cap.release()
cv2.destroyAllWindows()
