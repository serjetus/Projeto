from models import GetKeypoint
import numpy as np
import math
import cv2
from ultralytics import YOLO
modelpose= YOLO('yolov8n-pose.pt')


def color_detection_rgb(image):
    condition1 = (image[:, :, 0] > 95) & (image[:, :, 1] > 40) & (image[:, :, 2] > 20) & \
                 (np.max(image, axis=-1) - np.min(image, axis=-1) > 15) & \
                 (np.abs(image[:, :, 0] - image[:, :, 1]) > 15) & \
                 (image[:, :, 0] > image[:, :, 1]) & (image[:, :, 0] > image[:, :, 2])

    condition2 = (image[:, :, 0] > 220) & (image[:, :, 1] > 210) & (image[:, :, 2] > 170) & \
                 (np.abs(image[:, :, 0] - image[:, :, 1]) <= 15) & \
                 (image[:, :, 0] > image[:, :, 2]) & (image[:, :, 1] > image[:, :, 2])

    mask = condition1 | condition2
    result = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
    return result


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
        self.skin_segmentation = image
        self.height_pixels = 0

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

    def set_height(self, height):
        self.height_pixels = height

    def check_lost_track(self, fps, frame_count):
        return ((frame_count - self.lastFrame)/fps) >= 5

    def extract_caracteristcs(self):
        result_hsv = color_detection_hsv(self.image)
        result_ycrcb = color_detection_ycrcb(self.image)
        result_rgb = color_detection_rgb(self.image)
        final_result = cv2.bitwise_and(result_hsv, result_ycrcb)
        final_result = replace_non_black_pixels_with_white(final_result)
        blurred = cv2.GaussianBlur(final_result, (5, 5), 0)
        gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        kernel = np.ones((3, 3), np.uint8)
        blurred = cv2.dilate(blurred, kernel, 1)
        self.skin_segmentation = replace_non_white_pixels_with_black(blurred)
        #final_result = cv2.bitwise_and(result_hsv, cv2.bitwise_and(result_ycrcb, result_rgb))

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

            '''for x, y, confidence in list_re:
                if confidence > 0.6:
                    cv2.circle(blurred, (int(x), int(y)), 1, (0, 255, 0), -1)'''

            if list_re[1][2] and list_re[5][2] > 0.6:
                self.set_height(list_re[1][1] - list_re[5][1])

            if self.compare_circles(list_re[1][0], list_re[1][1]) and self.compare_circles(list_re[3][0], list_re[3][1]):
                if self.compare_circles(list_re[5][0]):
                    self.caracterics.append('MANGA LONGA')
                else:
                self.caracterics.append('REGATA')
            else:
                self.caracterics.append('CAMISA')
            if self.compare_circles(list_re[7][0], list_re[7][1]):
                self.caracterics.append('SHORTS')
            else:
                self.caracterics.append('CALÇA')

            print(self.caracterics)
            cv2.waitKey(0)

    def compare_circles(self, x, y):
        x = int(x)
        y = int(y)
        b, g, r = self.skin_segmentation[y, x] / 255.0
        print ((b, g, r) == (1.0, 1.0, 1.0))
        return (b, g, r) == (1.0, 1.0, 1.0)

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

    def getdistance(self, cx, cy, frame, fps):
        distance = math.hypot(cx - (int(self.x1+self.x2)/2), cy - (int(self.y1 + self.y2) / 2))
        sec = int((frame - self.lastFrame))
        return distance/sec if sec > 0 else distance

    def viewcenter(self):
        cv2.imshow('centro', self.centerpx)
        cv2.waitKey(0)
