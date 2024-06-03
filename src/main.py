import datetime
import math
import cv2
from ultralytics import YOLO
from people import People
from car import Car
from thread import WhatsAppThread


carro = None
model = YOLO("yolov8n.pt")
persons = []
personsT = []
centerParkX = int((215 + 506) / 2)
centerParkY = int((89 + 380) / 2)
centerparkGate_x = (10 + 10) / 2
centerparkGate_y = (10 + 10) / 2
stopedCars = []


def tracking(fps, pixels, frame, framecount, x1, x2, y1, y2, bcenterx, bcentery, pessoas):
    flag_2 = False
    for i in range(len(persons)):
        dist = persons[i].getdistance(bcenterx, bcentery, framecount, fps)
        if not flag_2 and dist < pixels:
            boxpeople = frame[y1:y2, x1:x2]
            persons[i].compare_bouding(boxpeople)
            persons[i].set_codinates(x1, x2, y1, y2)
            persons[i].set_lastframe(framecount)
            persons[i].reverse_track()
            flag_2 = True

    if not flag_2 and len(persons) < pessoas:
        boundingboxpeople = frame[y1:y2, x1:x2]
        person1 = People(boundingboxpeople, x1, x2, y1, y2, framecount)
        persons.append(person1)

        '''        for cod in range(len(persons)):
        if persons[cod].get_tracking():
            org = (persons[cod].get_cx(), persons[cod].get_cy() - 7)
            persons[cod].reverse_track()
            cv2.circle(frame, (bcenterx, bcentery), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(cod), org, 0, 1, (0, 0, 255), 2)'''

# telefone, roi1, roi2, roi3, limite, tempo


def extraction_process(fps, framecount):
    global persons, personsT
    match_flag = False
    for rmv in range(len(persons)):
        if persons[rmv].check_lost_track(fps, framecount):
            removed_person = persons.pop(rmv)
            removed_person.set_timedetection(datetime.datetime.now())
            print("EXTRAINDO")
            removed_person.extract_caracteristcs()
            print(removed_person.caracterics)
            match_flag = False
            for personC in personsT:
                print("COMPARANDO")
                print("", removed_person.caracterics, "= ", personC.caracterics)
                print("", removed_person.clothes_color, "= ", personC.clothes_color)
                match_flag = personC.caracterics == removed_person.caracterics and personC.clothes_color == removed_person.clothes_color
                if match_flag:
                    print("REDETECÇÂO")
                    personC.set_detections(personC.get_detections() + 1)
                    removed_person = None
                    break
            if not match_flag:
                print("ADICIONADO")
                personsT.append(removed_person)
                removed_person = None
            # personsT[len(personsT)-1].extract_caracteristcs() #a pessoa que saiu do rastreamento


def process(frame, framecount, fps, pixels, tempo_carro, telefone, tempo_pessoa, exibir_roi, exibir_pontos, rois):
    global carro, persons, personsT, centerParkX, centerParkY, centerparkGate_x, centerparkGate_y, stopedCars
    flag = False
    results = model(frame, verbose=False)
    for result in results:
        pessoas = sum(1 for elemento in result.boxes.data.tolist() if elemento[-1] == 0.0)
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1, y1, x2, y2, class_id = map(int, (x1, y1, x2, y2, class_id))
            bcenterx = int((x1 + x2) / 2)
            bcentery = int((y1 + y2) / 2)
            cv2.circle(frame, (bcenterx, bcentery), exibir_pontos, (0, 255, 0), -1)
            xr1, yr1, xr2, yr2 = 0, 0, 0, 0
            if exibir_roi:
                for roi_num, (xr1, yr1, xr2, yr2) in rois:
                    cv2.rectangle(frame, (xr1, yr1), (xr2, yr2), (0, 0, 255), 2)
                    cv2.circle(frame, (int((xr1 + xr2) / 2), int((yr1 + yr2) / 2)), 3, (0, 0, 255), -1)
                # cv2.rectangle(frame, (215, 89), (506, 380), (0, 0, 255), 1)
                # cv2.circle(frame, (centerParkX, centerParkY), 3, (0, 0, 255), -1)

            for roi_num, (xr1, yr1, xr2, yr2) in rois:
                center_x = int((xr1 + xr2) / 2)
                center_y = int((yr1 + yr2) / 2)
                if not flag:
                    flag = math.hypot(center_x - (int(x1 + x2) / 2), center_y - (int(y1 + y2) / 2)) < 30

            # flag = math.hypot(centerParkX - (int(x1 + x2) / 2), centerParkY - (int(y1 + y2) / 2)) < 30
            # flag_gate = math.hypot(centerparkGate_x - (int(x1 + x2) / 2), centerparkGate_y - (int(y1 + y2) / 2)) < 30

            extraction_process(fps, framecount)

            '''            if class_id == 2 and carro is not None and not flag:
                carro = None'''
            if class_id == 2 and carro is None and flag:
                carro = Car(frame[y1:y2, x1:x2], framecount, bcenterx, bcentery)
            else:
                if carro is not None:
                    if carro.getStopedTime(fps, framecount) >= tempo_carro and not carro.get_alerted():
                        if carro.get_alerted():
                            stopedCars.append(carro)
                        carro.viewimage(bcenterx, bcentery)
                        whatsapp_thread = WhatsAppThread(telefone, "./src/carro.jpg", "Carro estacionado", 2)
                        whatsapp_thread.start()

            if class_id == 0:
                if framecount < 1:
                    boundingboxpeople = frame[y1:y2, x1:x2]
                    person = People(boundingboxpeople, x1, x2, y1, y2, framecount)
                    persons.append(person)
                    print(persons)
                else:
                    tracking(fps, pixels, frame, framecount, x1, x2, y1, y2, bcenterx, bcentery, pessoas)
