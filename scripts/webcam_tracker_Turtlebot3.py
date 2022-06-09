#!/usr/bin/env python3

# This script demonstrates the object tracking portion of the TurtleBot-Follow-Person project.
# Running this script with the 'MobileNetSSD' files in the same directory will start the tracker
# and use the webcam for the camera feed.

import sys, datetime
import os
import rospy
import math
import cv2
import glob
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from time import sleep
from pynput import keyboard
import imutils

KILL = False

# function for registering key presses
def on_press(key):
	if key.char == 'q':
		global KILL
		KILL = True
		#print('Killing now')

def callback(data):
 
	# Used to convert between ROS and OpenCV images
	br = CvBridge()
	
	# Output debugging information to the terminal
	rospy.loginfo("receiving video frame")

	# load the serialized machine learning model for person detection
	net = cv2.dnn.readNetFromCaffe('/home/mai/Documents/follower_people/TurtleBot-Follow-Person/MobileNetSSD_deploy.prototxt.txt', '/home/mai/Documents/follower_people/TurtleBot-Follow-Person/MobileNetSSD_deploy.caffemodel')

	# initialize the list of class labels MobileNet SSD was trained to
	# detect (person is 15), then generate a set of bounding box colors for each class
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]
	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
	
	# initialize webcam
	# cap = cv2.VideoCapture(0)

	# # Take first frame
	# ret, frame = cap.read()
	frame = br.imgmsg_to_cv2(data)
	(h, w) = frame.shape[:2]

	# video writer setup
	fourcc = cv2.VideoWriter_fourcc('M','J','P','G') #Define the codec and create VideoWriter object
	out = cv2.VideoWriter('webcam_tracker.avi',fourcc, 20.0, (w,h))

	# capture next frame and convert to grayscale
	frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	# convert frame to a blob for object detection
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		object_type = detections[0,0,i,1]
		confidence = detections[0, 0, i, 2]
		if object_type == 15 and confidence > 0.2: # execute if confidence is high for person detection
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format('person',confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[int(object_type)], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[int(object_type)], 2)

	# write frame to video file and display			
	out.write(frame)
	frame = cv2.resize(frame, (1920//2,1080//2))
	cv2.imshow('Webcam Tracking', frame)

	if KILL:
		print("\nFinished")
		out.release()
		cv2.destroyAllWindows()
		exit()
	cv2.waitKey(1)
   



def receive_message():
 
  # Tells rospy the name of the node.
  # Anonymous = True makes sure the node has a unique name. Random
  # numbers are added to the end of the name. 
  rospy.init_node('video_sub_py', anonymous=True)
   
  # Node is subscribing to the video_frames topic
  rospy.Subscriber('/camera/rgb/image_raw', Image, callback)
 
  # spin() simply keeps python from exiting until this node is stopped
  rospy.spin()
 
  # Close down the video stream when done
  cv2.destroyAllWindows()
  


if __name__ == '__main__':
	listener = keyboard.Listener(on_press=on_press)
	listener.start()
	receive_message()
	exit()