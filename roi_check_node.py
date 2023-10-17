#! /usr/bin/env python3
# -*- coding:utf-8 -*-

# ==============================================================================
# Title: roi_check_node.py
# Description: roi check node
# Author: @GeonhaPark
# Date: 2022-10-19
# Version: 0.1
# Usage: roslaunch roi_check_node roi_check_node.launch
# ==============================================================================

import cv2
import numpy as np

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

cv_bridge = CvBridge()


class ROI_CHECK:
    def __init__(self, ref_image=None):
        rospy.init_node("color_filter", anonymous=True)
        # image BGR->HSV로 변환
        if ref_image is None:
            # 이미지 생성 (검은 배경)
            ref_image = np.ones((800, 400, 3), np.uint8)

            # 색 정의
            blue = np.array([[[255, 0, 0]]], dtype=np.uint8)
            green = np.array([[[0, 255, 0]]], dtype=np.uint8)
            red = np.array([[[0, 0, 255]]], dtype=np.uint8)
            yellow = np.array([[[0, 255, 255]]], dtype=np.uint8)
            lightblue = np.array([[[255, 255, 0]]], dtype=np.uint8)
            purple = np.array([[[255, 0, 255]]], dtype=np.uint8)
            white = np.array([[[255, 255, 255]]], dtype=np.uint8)

            ref_image[100:200, :] = green
            ref_image[200:300, :] = blue
            ref_image[300:400, :] = red
            ref_image[400:500, :] = yellow
            ref_image[500:600, :] = purple
            ref_image[600:700, :] = lightblue
            ref_image[700:800, :] = white

            red_hsv = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
            green_hsv = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
            blue_hsv = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
            yellow_hsv = cv2.cvtColor(yellow, cv2.COLOR_BGR2HSV)
            lightblue_hsv = cv2.cvtColor(lightblue, cv2.COLOR_BGR2HSV)
            purple_hsv = cv2.cvtColor(purple, cv2.COLOR_BGR2HSV)
            white_hsv = cv2.cvtColor(white, cv2.COLOR_BGR2HSV)

            print("green:    ", green_hsv)
            print("blue:     ", blue_hsv)
            print("red:      ", red_hsv)
            print("yellow:   ", yellow_hsv)
            print("purple:   ", purple_hsv)
            print("lightblue:", lightblue_hsv)
            print("white:    ", white_hsv)
        self.image = ref_image
        self.ref_image = ref_image
        self.editted_image = ref_image
        self.image_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.ref_image_hsv = self.image_hsv
        self.editted_image_hsv = self.image_hsv
        self.editted_tmp_image = ref_image
        self.lhue = 0
        self.lsat = 0
        self.lval = 0
        self.hhue = 0
        self.hsat = 0
        self.hval = 0
        self.update_color()
        pass

    def image_callback(self, image):
        cimg = cv_bridge.imgmsg_to_cv2(image, "bgr8")
        self.image = cimg

        ##########=====CHECK+++ROI====##########
        # transformation
        self.editted_tmp_image = cv2.resize(
            self.image, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_CUBIC
        )
        rows, cols, ch = self.editted_tmp_image.shape

        #### ADD LINES FOR pts1 ####
        dot1 = np.array([10, 80])
        dot2 = np.array([0, 130])
        dot3 = np.array([180, 80])
        dot4 = np.array([190, 130])
        cv2.circle(
            self.editted_tmp_image, (dot1[0], dot1[1]), radius=5, color=(255, 0, 0), thickness=-1
        )
        cv2.circle(
            self.editted_tmp_image, (dot2[0], dot2[1]), radius=5, color=(255, 0, 0), thickness=-1
        )
        cv2.circle(
            self.editted_tmp_image, (dot3[0], dot3[1]), radius=5, color=(255, 0, 0), thickness=-1
        )
        cv2.circle(
            self.editted_tmp_image, (dot4[0], dot4[1]), radius=5, color=(255, 0, 0), thickness=-1
        )
        cv2.line(
            self.editted_tmp_image,
            (dot1[0], dot1[1]),
            (dot2[0], dot2[1]),
            color=(255, 255, 255),
            thickness=1,
        )
        cv2.line(
            self.editted_tmp_image,
            (dot2[0], dot2[1]),
            (dot4[0], dot4[1]),
            color=(255, 255, 255),
            thickness=1,
        )
        cv2.line(
            self.editted_tmp_image,
            (dot4[0], dot4[1]),
            (dot3[0], dot3[1]),
            color=(255, 255, 255),
            thickness=1,
        )
        cv2.line(
            self.editted_tmp_image,
            (dot3[0], dot3[1]),
            (dot1[0], dot1[1]),
            color=(255, 255, 255),
            thickness=1,
        )
        #### ADD LINES ####

        pts1 = np.float32(
            [dot1, dot2, dot3, dot4]
        )  # [[15,60],[5,130],[180,60],[190,130]]
        pts2 = np.float32([[0, 0], [0, 300], [300, 0], [300, 300]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        # editted_img_size = (editted_tmp_img.shape[1], editted_tmp_img.shape[0])
        self.editted_image = cv2.warpPerspective(
            self.editted_tmp_image, M, (300, 300)
        )  # img_size
        self.editted_image_hsv = cv2.cvtColor(self.editted_image, cv2.COLOR_BGR2HSV)
        ########################################

        self.image_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.update_color()

    def update_color(self):
        self.lower = np.array([self.lhue, self.lsat, self.lval])
        self.upper = np.array([self.hhue, self.hsat, self.hval])
        self.mask = cv2.inRange(self.image_hsv, self.lower, self.upper)
        self.image_res = cv2.cvtColor(
            cv2.bitwise_and(self.image_hsv, self.image_hsv, mask=self.mask),
            cv2.COLOR_HSV2BGR,
        )
        self.ref_image_mask = cv2.inRange(self.ref_image_hsv, self.lower, self.upper)
        self.ref_image_res = cv2.cvtColor(
            cv2.bitwise_and(
                self.ref_image_hsv, self.ref_image_hsv, mask=self.ref_image_mask
            ),
            cv2.COLOR_HSV2BGR,
        )
        #######=====CHECK+++ROI====######
        lower_red = np.array([self.lhue, self.lsat, self.lval])
        upper_red = np.array([self.hhue, self.hsat, self.hval])
        lower_red2 = np.array([240, 42, 96])
        upper_red2 = np.array([255, 255, 255])

        maskr = cv2.inRange(self.editted_image_hsv, lower_red, upper_red)
        maskr2 = cv2.inRange(self.editted_image_hsv, lower_red2, upper_red2)
        maskr4 = cv2.bitwise_or(maskr2, maskr)
        # filter mask
        kernel = np.ones((7, 7), np.uint8)
        opening = cv2.morphologyEx(maskr4, cv2.MORPH_OPEN, kernel)
        rgb_yb2 = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        self.editted_image_res = rgb_yb2.copy()
        h, w = self.editted_image_res.shape
        search_top = int(1 * h / 4 + 20)
        search_bot = int(3 * h / 4 + 20)
        search_mid = int(w / 2)
        self.editted_image_res[0:search_top, 0:w] = 0
        self.editted_image_res[search_bot:h, 0:w] = 0
        #######=====CHECK+++ROI====######
        return

    def lhue_control(self, x):
        self.lhue = x
        self.update_color()
        return

    def lsat_control(self, x):
        self.lsat = x
        self.update_color()
        return

    def lval_control(self, x):
        self.lval = x
        self.update_color()
        return

    def hhue_control(self, x):
        self.hhue = x
        self.update_color()
        return

    def hsat_control(self, x):
        self.hsat = x
        self.update_color()
        return

    def hval_control(self, x):
        self.hval = x
        self.update_color()
        return

    def run(self):
        cv2.namedWindow("hsv filter trackbar")
        cv2.createTrackbar("lHue", "hsv filter trackbar", 0, 255, self.lhue_control)
        cv2.createTrackbar("lSat", "hsv filter trackbar", 0, 255, self.lsat_control)
        cv2.createTrackbar("lVal", "hsv filter trackbar", 0, 255, self.lval_control)
        cv2.createTrackbar("hHue", "hsv filter trackbar", 0, 255, self.hhue_control)
        cv2.createTrackbar("hSat", "hsv filter trackbar", 0, 255, self.hsat_control)
        cv2.createTrackbar("hVal", "hsv filter trackbar", 0, 255, self.hval_control)

        rate = rospy.Rate(10)  # Set a loop rate (10 Hz in this example)
        while not rospy.is_shutdown():
            # 사용자 입력 대기 (키보드 입력을 대기하고 창을 닫을 때까지 대기)

            # 이미지를 화면에 표시
            cv2.imshow("hsv filter trackbar", self.ref_image_res)
            cv2.imshow("image", self.image)
            # cv2.imshow("mask", self.mask)
            cv2.imshow("result", self.image_res)
            cv2.imshow("editted image", self.editted_tmp_image)
            cv2.imshow("editted image result", self.editted_image_res)

            pixel_count = cv2.countNonZero(self.mask)
            total_pixel_count = self.image.shape[0] * self.image.shape[1]
            area_ratio = pixel_count / total_pixel_count
            print(f"이미지 크기: {self.image.shape}")
            print(f"영역의 원본 대비 비율: {area_ratio:.2%}")
            if cv2.waitKey(1) & 0xFF == 27:
                break
            else:
                self.lhue = cv2.getTrackbarPos("lHue", "hsv filter trackbar")
                self.lsat = cv2.getTrackbarPos("lSat", "hsv filter trackbar")
                self.lval = cv2.getTrackbarPos("lVal", "hsv filter trackbar")
                self.hhue = cv2.getTrackbarPos("hHue", "hsv filter trackbar")
                self.hsat = cv2.getTrackbarPos("hSat", "hsv filter trackbar")
                self.hval = cv2.getTrackbarPos("hVal", "hsv filter trackbar")
                pass
            rate.sleep()

        cv2.destroyAllWindows()
        return


if __name__ == "__main__":
    img = cv2.imread("img_src/hsv_table.png", cv2.IMREAD_COLOR)
    color_filter_node = ROI_CHECK(img)
    image_sub = rospy.Subscriber(
        "/usb_cam/image_raw",
        Image,
        color_filter_node.image_callback,
        queue_size=1,
        buff_size=52428800,
    )
    color_filter_node.run()
