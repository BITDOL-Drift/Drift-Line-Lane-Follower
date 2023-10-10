#!/usr/bin/env python
# -*- coding:utf-8 -*-

#==============================================================================
# Title: lidar_avoidance_final_ref.py
# Description: ROS node for lidar avoidance _ REFERENCE
# Author: @GyJin-Kyung @Songhee-LEE @Takyoung-KIM
# Date: 2022-10-19
# Version: 0.1
# Usage: roslaunch lidar_avoidance_final_ref lidar_avoidance_final_ref.launch
# ROS node for lidar avoidance
#==============================================================================

# BEGIN ALL
import rospy
import os
import cv2
import cv_bridge
import numpy
import math
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from traffic_light_classifier.msg import traffic_light

global green_cnt

green_cnt = 0

err1_prev = 0 
err2_prev = 0


class Follower:
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()

        self.traffic_sub = rospy.Subscriber("/light_color", traffic_light, self.traffic_callback)
        self.lidar_sub = rospy.Subscriber("/scan_raw", LaserScan, self.lidar_callback)
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)

        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.image_pub = rospy.Publisher("/lane_image", Image, queue_size=1)
        self.image_debug_pub = rospy.Publisher("/lane_debug_image", Image, queue_size=1)

        self.twist = Twist()
        self.cmd_vel_pub.publish(self.twist)
        self.roi_center_pix = [0,0,0,0] # 1~3 index will be used

        self.dists = None
        self.traffic_color = 0

        ###------------------ 파라미터 수정 -----------------------###
        ##@@ 실행 모드
        self.debug = True

        ##@@ 라인검출 색 범위
        self.lower_red = numpy.array([143,75,53])
        self.upper_red = numpy.array([202,255,255])
        self.lower_red2 = numpy.array([0,75,53])
        self.upper_red2 = numpy.array([10,255,255])

        # ! linear.x, angular.z는 cilab_driver에서 1000이 곱해져 register값으로 적용된다. 0.001까지 유효하다.
        ##@@ 제어 파라미터
        self.K1 = 0.004
        self.D1 = 0.014

        # main
        self.K2 = 0.004
        self.D2 = 0.007

        self.obj_offset1 = 0#100
        self.obj_offset2 = 0
        ###======================================================###


    ## : /light_color 토픽에 따른 콜백함수 - 신호판별 node의 신호판단값 저장
    def traffic_callback(self, msg):
        global green_cnt
        self.traffic_color = msg.recognition_result

        if self.traffic_color == 1:
            green_cnt += 1
            # print(green_cnt)

    ## : /scan_raw 토픽에 따른 콜백함수 - lidar값 저장
    def lidar_callback(self, msg):
        # static offset
        angles = [x for x in range(-5, -90, -5)]  # -5, -90, -5
        print(angles)
        self.dists = [msg.ranges[x * 2] for x in angles]


    
    ## : /usb_cam/image_raw 토픽에 따른 콜백함수 - line tracking 실행
    def image_callback(self, msg):
        global perr, ptime, serr, dt
        global err1_prev, err2_prev

        # ------ take image and transform to hsv ------ #
        image0 = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        img = cv2.resize(image0, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_CUBIC)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # ============================================= #

        # ------------ extract red line -------------- #
        maskr = cv2.inRange(hsv, self.lower_red, self.upper_red)
        maskr2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        line_img = cv2.bitwise_or(maskr2, maskr)
        # ============================================= #

        # ------------ red obj noise filter ----------- #
        kernel = numpy.ones((3, 3), numpy.uint8)
        line_img = cv2.morphologyEx(line_img, cv2.MORPH_ERODE, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT)
        line_img = cv2.morphologyEx(line_img, cv2.MORPH_DILATE, kernel, iterations=2, borderType=cv2.BORDER_CONSTANT)
        # ============================================= #

        # --------------------------- 영역 정보 ---------------------------------------- #
        # ! 화면의 가장 아래는 차 범퍼로부터 11cm 떨어진 부분이다.
        # ! 1m 떨어진 바닥은 h*0.65 위치의 화면에 잡힌다. 이 위로는 관객이나 기타 사물에 영향을 받을 수 있다.

        # ============================================================================= #


        # ----------------------- calculate offset -------------------------- #
        if self.dists != None:
            lateral_count = 0
            for d in self.dists:
                if d < 0.65:  # 0.55
                    lateral_count += 1

            if lateral_count >= 1:
                # print("lateral_cnt : {}".format(lateral_count))
                roi1_offset = self.obj_offset1
                roi2_offset = self.obj_offset2
            else:
                roi1_offset = 0
                roi2_offset = 0
        else:
            roi1_offset = 0
            roi2_offset = 0
        # ============================================================================= #


        # ----------------------- calculate error -------------------------- #
        # h, w = line_img.shape # 144, 192
        h = 144
        w = 192
        w_mid = 96 # 120일 때 선을 거의 중앙에 둠, 근데 그럼 좌우 턴 최대값이 달라질 수 있음

        roi1_down = 144     # 4/4 * h
        roi1_up = 108       # 3/4 * h
        roi1_mid = 126

        roi2_down = 108     # 3/4 * h
        roi2_up = 72        # 2/4 * h
        roi2_mid = 90

        roi1 = line_img[roi1_up:roi1_down, 0:w]
        self.calc_center_point_idxavr(roi1, 1, 10)
        err1 = self.roi_center_pix[1] - roi1_offset - w_mid
        derr1 = err1 - err1_prev


        roi2 = line_img[roi2_up:roi2_down, 0:w]
        self.calc_center_point_idxavr(roi2, 2, 10)
        err2 = self.roi_center_pix[2] - roi2_offset - w_mid
        derr2 = err2 - err2_prev
        # ==================================================================== #


        # ------------------------ debug image make -------------------------- #
        if self.debug:
            cv2.circle(img, (w_mid, roi1_mid), 2, (0, 255, 255), 2)
            cv2.putText(img, str(err1), (w_mid, roi1_mid), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
            cv2.circle(img, (self.roi_center_pix[1], roi1_mid), 3, (0, 255, 0), 2) 
            cv2.circle(img, (self.roi_center_pix[1] - roi1_offset, roi1_mid), 6, (255, 0, 0), 2)
            
            cv2.circle(img, (w_mid, roi2_mid), 2, (0, 255, 255), 2) 
            cv2.putText(img, str(err2), (w_mid, roi2_mid), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
            cv2.circle(img, (self.roi_center_pix[2], roi2_mid), 3, (0, 255, 0), 2) 
            cv2.circle(img, (self.roi_center_pix[2] - roi2_offset, roi2_mid), 6, (255, 0, 0), 2)

            self.image_debug_pub.publish(self.bridge.cv2_to_imgmsg(img,encoding='bgr8'))
        # ==================================================================== #

        # ------------------ calculate velocity, angle ----------------------- #
        # @@ 조향값 범위 : -0.6~0.6, 속도값 범위 : -2.0~2.0
        # @@ error값은 -90~90 정도 범위를 가짐, 즉, K제어만 할 때는 K값이 0.007~0.006일 때 풍부한 제어

        ''' TUNING RECORD
        ang = err1 * self.K1 + derr1 * self.D1 : 
        sp=0.5, K1=0.007 no swing, very good
        sp=0.7, K1=0.007 little swing
        sp=0.8, K1=0.007 basic swing
        sp=1.0, K1=0.007 big swing

        sp=1.0, K1=0.004 커스텀코스1 못따라감, 대회코스는 따라갈 듯
        sp=1.0, K1=0.004 대회코스 곡률는 따라갈 듯, 지름 135cm정도의 곡선 가능, little swing
        sp=1.5, K1=0.004, swing 커짐.
        sp=1.0, K1=0.006  a lot swing
        sp=1.0, K1=0.006 D1=0.003  a little swing
        sp=1.0, K1=0.006 D1=0.006 little swing 
        sp=1.0, K1=0.006 D1=0.01  almost no swing !!
        sp=1.0, K1=0.006 D1=0.02  basic swing
        sp=1.0, K1=0.008 D1=0.01  big swing
        sp=1.0, K1=0.007 D1=0.012  basic swing
        sp=1.0, K1=0.007 D1=0.015  basic swing
        sp=1.0, K1=0.005 D1=0.008  no swing !!!

        sp=1.5, K1=0.004 D1=0.010  no swing !!! , 장애물 낫벧

        K1=0.003이하로 할 시 135cm 지름 턴 문제있음
        sp=1.8, K1=0.004 D1=0.010  basic swing 
        sp=1.8, K1=0.004 D1=0.013  basic swing
        sp=1.8, K1=0.004 D1=0.016  a lot swing
        sp=1.8, K1=0.004 D1=0.012  basic swing
        sp=1.8, K1=0.004 D1=0.008  basic swing

        sp=1.5, K1=0.004 D1=0.010  little swing
        sp=1.5, K1=0.004 D1=0.012  little swing

        sp=1.8, K1=0.004 D1=0.012  basic swing
        sp=2, K1=0.004 D1=0.012  basic swing
        sp=2, K1=0.004 D1=0.015  basic- swing
        sp=2, K1=0.004 D1=0.020  big swing
        sp=2, K1=0.004 D1=0.014  basic- swing

        ang = err2 * self.K2 + derr2 * self.D2 : 
        sp=2, K2=0.004 D2=0.14  little swing
        sp=2, K2=0.004 D2=0.18  little swing but crash
        sp=2, K2=0.004 D2=0.12  almost no swing
        sp=2, K2=0.004 D2=0.10  no swing !!!
        sp=2, K2=0.004 D2=0.08  no swing at all !!!!!!
        sp=2, K2=0.004 D2=0.06  no swing at all !!!!!!
        sp=2, K2=0.004 D2=0.01  a lot swing 
        sp=2, K2=0.004 D2=0.04  little swing 
        sp=2, K2=0.004 D2=0.07  no swing at all !!!!!! - perfect 

        '''

        # if green_cnt >= 1:
        if True:
            speed = 2
            # ang = err1 * self.K1 + derr1 * self.D1
            ang = err2 * self.K2 + derr2 * self.D2


            
            self.twist.linear.x = speed
            self.twist.angular.z = -ang
            
        # ==================================================================== #
        

        # ------------------------ message publish --------------------------- #
        err1_prev = err1
        err2_prev = err2

        self.cmd_vel_pub.publish(self.twist)
        output_img = self.bridge.cv2_to_imgmsg(line_img)
        self.image_pub.publish(output_img)
        # ==================================================================== #

    ## : def image_callback(self, msg) 함수에서 line의 중심점을 찾기 위해 사용
    # 이 함수는 모든 픽셀값에 대한 평균으로 정확한 평균 위치를 찾아주진 않지만 
    # 중위값을 계산함으로써 위 함수보다 빠르다. 또 노이즈 제거 효과도 있어서 calc_center_point_valavr보다 오히려 성능이 더 좋음.
    def calc_center_point_idxavr(self, bframe, roi_num, threshold): #-> int
        
        # 이미지의 가로 인덱스별 픽셀 갯수 ndarray반환 : bframe_hist는 1차 ndarray 타입
        bframe_hist = numpy.sum(bframe,axis=0) 
        
        # 픽셀갯수가 threshold값을 넘는 ndarray의 index 반환, 1차 ndarray를 인수로 받았기 때문에 출력은 (1차 ndarray,) 튜플로 반환됨, 그래서 [0]을 사용
        pixel_idx_list = (numpy.where(bframe_hist > threshold))[0] 

        if len(pixel_idx_list) > 4 :
            # 즉, [가로 인덱스마다 세로로 합친 픽셀 갯수가 threshold값을 넘어가는 가로 인덱스값들](=[pixel_idx_list])의 평균을 반환
            self.roi_center_pix[roi_num] = int(numpy.average(pixel_idx_list))
            
# END CONTROL #


rospy.init_node("follower")
follower = Follower()
rospy.spin()
# END ALL
