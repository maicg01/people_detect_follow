#!/usr/bin/env python3

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
 
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)

def draw_boxes(frame, boxes, color=(0,255,0)):
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
    return frame

def resize_image(image, size_limit=500.0):
    max_size = max(image.shape[0], image.shape[1])
    if max_size > size_limit:
        scale = size_limit / max_size
        _img = cv2.resize(image, None, fx=scale, fy=scale)
        return _img
    return image

class FaceDetector():

    def __init__(self, cascPath="./haarcascade_frontalface_default.xml"):
        self.faceCascade = cv2.CascadeClassifier(cascPath)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces

class FaceTracker():
    
    def __init__(self, frame, face):
        (x,y,w,h) = face
        self.face = (x,y,w,h)
        # Arbitrarily picked KCF tracking
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, self.face)
    
    def update(self, frame):
        _, self.face = self.tracker.update(frame)
        return self.face

class Controller():
    
    def __init__(self, event_interval=6):
        self.event_interval = event_interval
        self.last_event = datetime.datetime.now()

    def trigger(self):
        """Return True if should trigger event"""
        return self.get_seconds_since() > self.event_interval
    
    def get_seconds_since(self):
        current = datetime.datetime.now()
        seconds = (current - self.last_event).seconds
        return seconds

    def reset(self):
        self.last_event = datetime.datetime.now()

class Pipeline():

    def __init__(self, event_interval=6):
        self.controller = Controller(event_interval=event_interval)    
        self.detector = FaceDetector()
        self.trackers = []
    
    def detect_and_track(self, frame):
        # get faces 
        faces = self.detector.detect(frame)

        # reset timer
        self.controller.reset()

        # get trackers
        self.trackers = [FaceTracker(frame, face) for face in faces]

        # return state = True for new boxes
        # if no faces detected, faces will be a tuple.
        new = type(faces) is not tuple

        return faces, new
    
    def track(self, frame):
        boxes = [t.update(frame) for t in self.trackers]
        # return state = False for existing boxes only
        return boxes, False
    
    def boxes_for_frame(self, frame):
        if self.controller.trigger():
            return self.detect_and_track(frame)
        else:
            return self.track(frame)

def event():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", "-i", type=int,
        action='store',
        default=6,
        help='Detection interval in seconds, default=6')

    args = parser.parse_args()
    return args.interval


def callback(data):
    br = CvBridge()
    rospy.loginfo("receiving video frame")
    # init detection pipeline
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", "-i", type=int,
        action='store',
        default=6,
        help='Detection interval in seconds, default=6')

    args = parser.parse_args()
    event_interval=args.interval
    pipeline = Pipeline(event_interval=event_interval)

    # hot start detection
    # read some frames to get first detection
    faces = ()
    detected = False
    if not detected:
        frame = br.imgmsg_to_cv2(data)
        faces, detected = pipeline.detect_and_track(frame)
        print("hot start; ", faces, type(faces), "size: ", np.array(faces).size)

    draw_boxes(frame, faces)
    
    ##
    ## main loop
    ##

    # Capture frame-by-frame
    frame = br.imgmsg_to_cv2(data)

    # update pipeline
    boxes, detected_new = pipeline.boxes_for_frame(frame)

    # logging
    state = "DETECTOR" if detected_new else "TRACKING"
    print("[%s] boxes: %s" % (state, boxes))

    # update screen
    color = GREEN if detected_new else BLUE
    draw_boxes(frame, boxes, color)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # quit
    cv2.waitKey(1)
      
def receive_message():
 
    rospy.init_node('video_sub_py', anonymous=True)
   
    rospy.Subscriber('/camera/rgb/image_raw', Image, callback)
 
    rospy.spin()

    cv2.destroyAllWindows()
  

if __name__ == '__main__':
    receive_message()
