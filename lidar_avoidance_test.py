#!/usr/bin/env python
# BEGIN ALL
import rospy
import cv2
import cv_bridge
import numpy as np
import math
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

global perr, ptime, serr, dt, move, ray_angle
perr = 0
ptime = 0
serr = 0
dt = 0
move = False
angle_step_deg = 20

# ros에서 이미지를 받아와서 트랙바를 사용하는 법
# https://stackoverflow.com/questions/51248709/frozen-black-image-and-trackbar-while-using-opencv-and-ros

class Follower:
	def __init__(self):
		self.bridge = cv_bridge.CvBridge()
		self.image_sub = rospy.Subscriber('/usb_cam/image_raw',	Image, self.image_callback)
		self.lidar_sub = rospy.Subscriber('/scan_raw', LaserScan, self.lidar_callback)
		self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
		self.image_pub = rospy.Publisher('/lane_image', Image, queue_size=1)
		self.twist = Twist()
		self.ray_angle = [x for x in range(angle_step_deg, 180, angle_step_deg)]
		self.dists = None

		self.cmd_vel_pub.publish(self.twist)

		cv2.namedWindow('HSV_settings')

		cv2.createTrackbar('H_MAX', 'HSV_settings', 0, 255, self.onChange)
		cv2.setTrackbarPos('H_MAX', 'HSV_settings', 255)
		cv2.createTrackbar('H_MIN', 'HSV_settings', 0, 255, self.onChange)
		cv2.setTrackbarPos('H_MIN', 'HSV_settings', 0)
		cv2.createTrackbar('S_MAX', 'HSV_settings', 0, 255, self.onChange)
		cv2.setTrackbarPos('S_MAX', 'HSV_settings', 255)
		cv2.createTrackbar('S_MIN', 'HSV_settings', 0, 255, self.onChange)
		cv2.setTrackbarPos('S_MIN', 'HSV_settings', 0)
		cv2.createTrackbar('V_MAX', 'HSV_settings', 0, 255, self.onChange)
		cv2.setTrackbarPos('V_MAX', 'HSV_settings', 255)
		cv2.createTrackbar('V_MIN', 'HSV_settings', 0, 255, self.onChange)
		cv2.setTrackbarPos('V_MIN', 'HSV_settings', 0)


	def onChange(self, pos):
		pass

	def lidar_callback(self, msg):
		# get lidar distance at ray_angle in degree
		# dynamic offset
		# angles = [(x - 90) % 360 for x in self.ray_angle]
		# self.dists = [msg.ranges[x*2] for x in angles]
		# self.dists = list(map(lambda x: 0.1 if x == float('inf') else x, self.dists))
		# self.dists = list(map(lambda x: 0.5 if x >= 0.5 else x, self.dists))

		# static offset
		angles = [x for x in range(-10, -90, -5)]
		self.dists = [msg.ranges[x*2] for x in angles]
		


	def get_obstacle_threshold(self):
		if self.dists == None:
			return 0

		# dynamic offset
		# lateral_dists = [dist * numpy.cos(numpy.deg2rad(theta)) for dist, theta in zip(self.dists, self.ray_angle)]

		# static offset
		lateral_count = 0
		for d in self.dists:
			if d < 0.5:
				lateral_count += 1
		if lateral_count >= 1:
			print("lateral_cnt :{}".format(lateral_count))
			return 120
		else:
			return 0

		# dynamic offset
		# return sum(lateral_dists)

	def image_callback(self, msg):
		global perr, ptime, serr, dt
		image0 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

		# transformation
		img = cv2.resize(image0, None, fx=0.6, fy=0.6,
						 interpolation=cv2.INTER_CUBIC)
		#print img.shape
		rows, cols, ch = img.shape
		
		'''
		pts1 = numpy.float32([[30, 80], [20, 130], [160, 80], [170, 130]])
		pts2 = numpy.float32([[0, 0], [0, 300], [300, 0], [300, 300]])

		M = cv2.getPerspectiveTransform(pts1, pts2)
		img_size = (img.shape[1], img.shape[0])
		image = cv2.warpPerspective(img, M, (300, 300))  # img_size
		'''

		image = img
		
		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

		H_MAX = cv2.getTrackbarPos('H_MAX', 'HSV_settings')
		H_MIN = cv2.getTrackbarPos('H_MIN', 'HSV_settings')
		S_MAX = cv2.getTrackbarPos('S_MAX', 'HSV_settings')
		S_MIN = cv2.getTrackbarPos('S_MIN', 'HSV_settings')
		V_MAX = cv2.getTrackbarPos('V_MAX', 'HSV_settings')
		V_MIN = cv2.getTrackbarPos('V_MIN', 'HSV_settings')
		lower = np.array([H_MIN, S_MIN, V_MIN])
		higher = np.array([H_MAX, S_MAX, V_MAX])
		Gmask = cv2.inRange(hsv, lower, higher)
		G = cv2.bitwise_and(image, image, mask = Gmask)


		cv2.imshow('HSV_settings',G)
		k = cv2.waitKey(1) & 0xFF
		if k == 27:
			exit(2)
		


rospy.init_node('follower')
follower = Follower()
rospy.spin()
# END ALL
