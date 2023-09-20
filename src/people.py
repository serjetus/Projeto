import cv2


class People:
    caracterics = []
    detections = 0
    detections_time = []
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

    def show_cordinates(self):
        print("X1:", self.x1, "X2:", self.x2, "Y1:", self.y1, "Y2:", self.y2)

    def compare_coordinates(self, x1, x2, y1, y2):
        return self.x1 == x1 and self.x2 == x2 and self.y1 == y1 and self.y2 == y2

    def as_moved(self, x1, x2, y1, y2, frame):
        if self.lastFrame - frame > 10:
            distx = self.getdist_xmove()
            increment = distx * (self.lastFrame - frame) / 20
            distx += increment
            disty = self.getdist_ymove()
            increment = disty * (self.lastFrame - frame) / 20
            disty += increment

            if (x1 <= self.x2 + distx <= x2 or x2 >= self.x1 - distx >= x1) and (
                    y1 <= self.y2 + disty <= y2 or y2 >= self.y1 - disty >= y1):
                return True
            else:
                return False



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
