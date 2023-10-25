import cv2

class Car:
    def __init__(self, image, frame):
        self.image = image
        self.frame = frame

    def set_image(self, image):
        self.image = image

    def set_frame(self, value):
        self.frame = value

    def get_image(self):
        return self.image

    def get_frame(self):
        return self.frame

    def viewimage(self):
        cv2.imshow('Carro Estacionado em frente a garagem', self.image)
        cv2.waitKey(0)
