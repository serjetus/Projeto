import cv2

class Car:
    alerted = False
    def __init__(self, image, frame, centerx, centery):
        self.image = image
        self.frame = frame
        self.centerX = centerx
        self.centerY = centery

    def set_image(self, image):
        self.image = image

    def set_frame(self, value):
        self.frame = value

    def get_image(self):
        return self.image

    def get_frame(self):
        return self.frame

    def get_alerted(self):
        return self.alerted

    def viewimage(self, centerx, centery):
        cv2.imshow('Carro Estacionado em frente a garagem', self.image)
        print("Carro Estacionado em frente a garagem")
        self.alerted = True
        #cv2.waitKey(0)

    def getStopedTime(self, fps, frame):
        return (frame-self.frame)/fps

