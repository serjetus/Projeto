from models import GetKeypoint
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from ultralytics import YOLO

modelpose = YOLO('yolov8n-pose.pt')


def color_detection_ycrcb(image):
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    mask = (
            (ycrcb_image[:, :, 1] <= 1.5862 * ycrcb_image[:, :, 2] + 20) &
            (ycrcb_image[:, :, 1] >= 0.3448 * ycrcb_image[:, :, 2] + 76.2069) &
            (ycrcb_image[:, :, 1] >= -4.5652 * ycrcb_image[:, :, 2] + 234.5652) &
            (ycrcb_image[:, :, 1] <= -1.15 * ycrcb_image[:, :, 2] + 301.75) &
            (ycrcb_image[:, :, 1] <= -2.2857 * ycrcb_image[:, :, 2] + 432.85)
    )
    result = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
    return result


def color_detection_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, (0, 25, 25), (25, 255, 255)) | cv2.inRange(hsv_image, (230, 25, 25), (255, 255, 255))
    result = cv2.bitwise_and(image, image, mask=mask)
    return result


def replace_non_black_pixels_with_white(image):
    non_black_mask = cv2.inRange(image, (1, 1, 1), (255, 255, 255))
    result = image.copy()
    result[non_black_mask > 0] = [255, 255, 255]
    return result


def replace_non_white_pixels_with_black(image):
    non_white_mask = cv2.inRange(image, (0, 0, 0), (254, 254, 254))
    result = image.copy()
    result[non_white_mask > 0] = [0, 0, 0]
    return result


def most_frequent_color(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calcula os histogramas para os canais HSV
    hist_hue = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
    hist_sat = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
    hist_val = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

    # Encontra o valor mais frequente para cada canal
    dominant_hue = np.argmax(hist_hue)
    dominant_sat = np.argmax(hist_sat)
    dominant_val = np.argmax(hist_val)

    dominant_color_hsv = [dominant_hue, dominant_sat, dominant_val]

    return dominant_color_hsv


class People:
    tracking = False

    def __init__(self, image, x1, x2, y1, y2, frame):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.lastFrame = frame
        self.image = image
        self.skin_segmentation = image
        self.height_pixels = 0
        self.caracterics = []
        self.clothes_color = []
        self.detections = 1
        self.detections_time = []
        self.alerted = False

    def get_alerted(self):
        return self.alerted

    def set_alerted(self, value):
        self.alerted = value

    def getdist_xmove(self):
        return self.x2 - self.x1

    def getdist_ymove(self):
        return self.y2 - self.y1

    def reverse_track(self):
        self.tracking = not self.tracking

    def get_tracking(self):
        return self.tracking

    def compare_bouding(self, image):
        altura1, largura1, canais1 = self.image.shape
        altura2, largura2, canais2 = image.shape
        if altura2 > altura1:
            self.image = image

    def viewimage(self):
        cv2.imshow('pessoa', self.image)
        cv2.waitKey(0)

    def set_codinates(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def set_timedetection(self, time):
        self.detections_time.append(time)

    def set_height(self, height):
        self.height_pixels = height

    def check_lost_track(self, fps, frame_count):
        return ((frame_count - self.lastFrame) / fps) >= 5

    def extract_caracteristcs(self):
        result_hsv = color_detection_hsv(self.image)
        result_ycrcb = color_detection_ycrcb(self.image)
        final_result = cv2.bitwise_and(result_hsv, result_ycrcb)
        final_result = replace_non_black_pixels_with_white(final_result)
        kernel = np.ones((3, 3), np.uint8)
        blurred = cv2.erode(final_result, kernel, 1)
        blurred = cv2.dilate(blurred, kernel, 1)
        blurred = cv2.dilate(blurred, kernel, 1)
        blurred = cv2.dilate(blurred, kernel, 1)
        self.skin_segmentation = blurred


        results = modelpose(source=self.image, )
        get_name = GetKeypoint()
        for keypoint in results:
            results_keypoint = keypoint.keypoints.data.tolist()
            list_re = [
                list(results_keypoint[0][get_name.LEFT_SHOULDER]),
                list(results_keypoint[0][get_name.RIGHT_SHOULDER]),
                list(results_keypoint[0][get_name.LEFT_ELBOW]),
                list(results_keypoint[0][get_name.RIGHT_ELBOW]),
                list(results_keypoint[0][get_name.LEFT_WRIST]),
                list(results_keypoint[0][get_name.RIGHT_WRIST]),
                list(results_keypoint[0][get_name.LEFT_KNEE]),
                list(results_keypoint[0][get_name.RIGHT_KNEE]),
                list(results_keypoint[0][get_name.LEFT_ANKLE]),
                list(results_keypoint[0][get_name.RIGHT_ANKLE]),
                list(results_keypoint[0][get_name.RIGHT_HIP]),
                list(results_keypoint[0][get_name.LEFT_HIP])
            ]
            if list_re[0][2] and list_re[5][2] > 0.6:
                self.set_height(list_re[1][1] - list_re[5][1])
            '''            for x, y, conf in list_re:
                cv2.circle(self.skin_segmentation, (int(x), int(y)), 1, (0, 255, 0), -1)
                cv2.circle(self.image, (int(x), int(y)), 1, (0, 255, 0), -1)
            cv2.imwrite("Seg_pontos.png", self.skin_segmentation)
            cv2.imwrite("pontos.png", self.image)'''

            OE = self.compare_circles(list_re[0][0], list_re[0][1])
            OD = self.compare_circles(list_re[1][0], list_re[1][1])
            CE = self.compare_circles(list_re[2][0], list_re[2][1])
            CD = self.compare_circles(list_re[3][0], list_re[3][1])

            if OE and OD:
                self.caracterics.append("REGATA")
            elif not (OE or OD):  # Nenhum ombro é branco
                if CE and CD:
                    self.caracterics.append("CAMISA")
                elif not (CE or CD):  # Nenhum cotovelo é branco
                    self.caracterics.append("MANGA LONGA")
                else:
                    dominant_point = list_re[2] if list_re[2][2] >= list_re[3][2] else list_re[3]
                    if self.compare_circles(dominant_point[0], dominant_point[1]):
                        self.caracterics.append("CAMISA")
            else:
                dominant_point = list_re[0] if list_re[0][2] >= list_re[1][2] else list_re[1]
                if self.compare_circles(dominant_point[0], dominant_point[1]):
                    self.caracterics.append("REGATA")

            JE = self.compare_circles(list_re[6][0], list_re[6][1])
            JD = self.compare_circles(list_re[7][0], list_re[7][1])
            TE = self.compare_circles(list_re[8][0], list_re[8][1])
            TD = self.compare_circles(list_re[9][0], list_re[9][1])

            if JE and JD or TE and TD:
                self.caracterics.append('SHORTS')
            else:
                if not (JE or JD and TE or TD):
                    self.caracterics.append("CALÇA")
                else:
                    if TE or TD:
                        dominant_point = list_re[8] if list_re[8][2] >= list_re[9][2] else list_re[9]
                        if self.compare_circles(dominant_point[0], dominant_point[1]):
                            self.caracterics.append("SHORTS")
                    elif JE or JD:
                        dominant_point = list_re[6] if list_re[6][2] >= list_re[7][2] else list_re[7]
                        if self.compare_circles(dominant_point[0], dominant_point[1]):
                            self.caracterics.append("SHORTS")

            '''            flag = False
            if not flag:
                if self.compare_circles(list_re[7][0], list_re[7][1] + 5):
                    self.caracterics.append('SHORTS')
                else:
                    self.caracterics.append('CALÇA')
                flag = True

            if not flag:
                if self.compare_circles(list_re[6][0], list_re[6][1] + 5):
                    self.caracterics.append('SHORTS')
                else:
                    self.caracterics.append('CALÇA')'''

            if list_re[1][2] > list_re[0][2]:  # qual ombro esta mais visivel
                shirt_image = self.image[int(list_re[1][1]):int(list_re[10][1]), int(list_re[1][0]):int(list_re[10][0]), :]
            else:
                shirt_image = self.image[int(list_re[0][1]):int(list_re[11][1]), int(list_re[0][0]):int(list_re[11][0]), :]
            '''            cv2.imwrite("camisa.jpg", self.image)
            cv2.imwrite("C_cortada.jpg", shirt_image)'''

            color = most_frequent_color(shirt_image)
            self.clothes_color.append(color)
            if list_re[10][2] > list_re[11][2]:
                legs_image = self.image[int(list_re[6][1]):int(list_re[6][1]) + 7, int(list_re[6][0]):int(list_re[6][0]) + 7, :]
            else:
                legs_image = self.image[int(list_re[7][1]):int(list_re[7][1]) + 7, int(list_re[7][0]):int(list_re[7][0]) + 7, :]

            #cv2.imwrite("camisa.jpg", self.image)
            color = most_frequent_color(legs_image)
            self.clothes_color.append(color)

            # cv2.waitKey(0)

    def compare_circles(self, x, y):
        x = int(x)
        y = int(y)
        b, g, r = self.skin_segmentation[y, x] / 255.0
        return (b, g, r) == (1.0, 1.0, 1.0)

    def set_image(self, image):
        self.image = None
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

    def set_detections(self, value):
        self.detections = value

    def get_detections(self):
        return self.detections

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
        return int((self.x1 + self.x2) / 2)

    def get_cy(self):
        return int((self.y1 + self.y2) / 2)

    def getdistance(self, cx, cy, frame, fps):
        distance = math.hypot(cx - (int(self.x1 + self.x2) / 2), cy - (int(self.y1 + self.y2) / 2))
        sec = int((frame - self.lastFrame))
        return distance / sec if sec > 0 else distance

    def getstopedtime(self, fps, frame):
        return (frame - self.lastFrame) / fps

    def save_image(self):
        cv2.imwrite("pessoa.jpg", self.image)

    def viewcenter(self):
        cv2.imshow('centro', self.centerpx)
        cv2.waitKey(0)
