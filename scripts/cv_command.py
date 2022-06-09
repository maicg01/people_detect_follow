import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist

#print(cv2.__version__)

# load the serialized machine learning model for person detection
global net
net = cv2.dnn.readNetFromCaffe('/home/mai/Documents/follower_people/TurtleBot-Follow-Person/MobileNetSSD_deploy.prototxt.txt', '/home/mai/Documents/follower_people/TurtleBot-Follow-Person/MobileNetSSD_deploy.caffemodel')

global counter
global count_max
counter = 0
count_max = 400

# video writer setup
global fourcc
global out
fourcc = cv2.VideoWriter_fourcc('M','J','P','G') #Define the codec and create VideoWriter object
out = cv2.VideoWriter('output_dnn.avi',fourcc, 20.0, (640,480))

class CVControl:
	def __init__(self):
		self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
		self.cmd = Twist()

		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.img_callback)

	def img_callback(self, data):
		global counter
		global net
		global out
		global count_max
		counter += 1

		CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			"sofa", "train", "tvmonitor"]

		COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data)
		
		except CvBridgeError as e:
			print(e)
			
		cv_gray = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
		frame = cv_image
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

		# pass the blob through the network and obtain the detections and
		# predictions

		net.setInput(blob)
		detections = net.forward()
		big_area = 0
		big_center = 320
		detected = 0

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			print(i)
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
				cv2.rectangle(frame, (startX, startY), (endX, endY),[0,0,255], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)
				# get bounding box center and area used for commanding the TurtleBot to follow a person
				rect_center = int((startX+endX)/2)
				rect_area = (endX-startX)*(endY-startY)
				detected = 1
				if rect_area > big_area: # big_area and big_center are used so that the TurtleBot always tracks the closest person
					big_area = rect_area
					big_center = rect_center
		
		# if a person is detected, send commands to approach or move away from them depending on proximity
		if detected:
			#print(rect_center)
			#print(rect_area)
			if big_area > 10000: # Execute if the person is within a reasonable range
				target_center = 320
				target_area = 150000
				# proportional controller for the TurtleBot based on the persons position in the frame and how far they are away
				kr = .002
				w = -kr*(big_center - target_center)
				kt = 0.0000045
				v = -kt*(big_area - target_area)
				maxv = 0.25 # Set a maximum velocity for the Turtlebot
				v = np.max([-maxv, v])
				v = np.min([maxv, v])
				#print(v)
				# Send Velocity command to turtlebot
				self.send_command(v, w)

		# Write frames to a video file for up to count_max frames
		if counter < count_max:
			out.write(frame)
			print(counter)

		if counter == count_max:
			out.release()
			print('made video')
	
	# cv2.imshow("Image window", frame)
	# cv2.waitKey(3)

	def send_command(self, v, w):
		# Put v, w commands into Twist message
		self.cmd.linear.x = v
		self.cmd.angular.z = w

		rospy.loginfo("public control vel")
		# Publish Twist command
		self.cmd_pub.publish(self.cmd)


def main():
	ctrl = CVControl()
	rospy.init_node('control_vel')
	try:
		rospy.spin()
	
	except KeyboardInterrupt:
		print("Shutting down")
		cv2.destroyAllWindows()

if __name__ == '__main__':
	main()


