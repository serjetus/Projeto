import os
import random

import cv2
from ultralytics import YOLO
from people import People

video = os.path.join('.', 'videos', 'people.mp4')
Video_outpath = os.path.join('.', 'rastreado.mp4')
cords = []
video_cap = cv2.VideoCapture(video)
ret, frame = video_cap.read()
'''cv2.VideoWriter_fourcc(*'MP4V')'''
video_capture_output = cv2.VideoWriter(Video_outpath, 0x7634706d, video_cap.get(cv2.CAP_PROP_FPS),
                                       (640, 380))

model = YOLO("yolov8n.pt")
color = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

persons = []
frameCount = 0
detection_threshold = 0.5

while ret:
    results = model(frame)
    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            print("Frame: ", frameCount)
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold and class_id == 0:
                if len(persons) == 0:
                    boundingBoxPeople = frame[y1:y2, x1:x2]
                    person = People(boundingBoxPeople, x1, x2, y1, y2, frameCount)
                    persons.append(person)
                    persons[0].show_cordinates()
                else:
                    flag = False
                    for i in range(len(persons)):
                        if not flag:
                            thesame = persons[i].compare_coordinates(x1, x2, y1, y2)
                            if persons[i].as_moved(x1, x2, y1, y2, frameCount) and thesame:
                                print("a pessoa", i, " se moveu")
                                persons[i].set_codinates(x1, x2, y1, y2)
                                persons[i].set_lastframe(frameCount)
                                flag = True
                    if not flag:
                        '''é uma nova pessoa que ainda não esta na lista'''
                        boundingBoxPeople = frame[y1:y2, x1:x2]
                        person = People(boundingBoxPeople, x1, x2, y1, y2, frameCount)
                        persons.append(person)
                        '''filename = f'pessoa_{len(cords) - 1}.jpg'
                        cv2.imwrite('src/BoundingBoxPrints/'+filename, boundingBoxPeople)'''

                detections.append([x1, y1, x2, y2, score])
                random_color = color[random.randint(0, len(color) - 1)]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 3)

    '''    cv2.imshow('teste', frame)
    cv2.waitKey(5)'''
    video_capture_output.write(frame)
    frameCount += 1
    ret, frame = video_cap.read()

video_cap.release()
video_capture_output.release()
cv2.destroyAllWindows()
