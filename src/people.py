import math

import cv2


class People:
    caracterics = []
    detections = 0
    detections_time = []
    tracking = False

    def __init__(self, image, x1, x2, y1, y2, frame):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.lastFrame = frame
        self.image = image

    def getdist_xmove(self):
        return self.x2 - self.x1

    def getdist_ymove(self):
        return self.y2 - self.y1

    def reverse_track(self):
        self.tracking = not self.tracking

    def get_tracking(self):
        return self.tracking

    def viewimage(self):
        cv2.imshow('pessoa', self.image)
        cv2.waitKey(10)

    def set_codinates(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def set_image(self, image):
        self.image = image

    def set_lastframe(self, value):
        self.lastFrame = value

    def set_x1(self, value):
        self.x1 = value

    def set_x2(self, value):
        self.x2 = value

    def set_y1(self, value):
        self.y1 = value

    def set_y2(self, value):
        self.y2 = value

    def get_image(self):
        return self.image

    def get_lastframe(self):
        return self.lastFrame

    def get_x1(self):
        return self.x1

    def get_x2(self):
        return self.x2

    def get_y1(self):
        return self.y1

    def get_y2(self):
        return self.y2

    def get_cx(self):
        return int((self.x1 + self.x2)/2)

    def get_cy(self):
        return int((self.y1 + self.y2)/2)

    def getdistance(self, cx, cy, frame):
        distance = math.hypot(cx - (int(self.x1+self.x2)/2), cy - (int(self.y1 + self.y2) / 2))
        frames = int((frame - self.lastFrame)/10)
        return distance/frames if frames > 0 else distance
