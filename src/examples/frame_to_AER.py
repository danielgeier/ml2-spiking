#!/usr/bin/env python
# PKG = 'dvs_simulation'
import roslib# ; roslib.load_manifest(PKG)

import rospy
import sys
import cv2
import std_msgs
# from dvs_msgs.msg import Event
# from dvs_msgs.msg import EventArray
from sensor_msgs.msg import Image, CameraInfo
from rospy.numpy_msg import numpy_msg
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

DVS_WIDTH = rospy.get_param('dvs_width', 128)
DVS_HEIGHT = rospy.get_param('dvs_height', 128)

class frameToAERConverter():
    def __init__(self):
        self.node_name = "frameToAER"
        rospy.init_node(self.node_name)

        rospy.on_shutdown(self.cleanup)

        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber("/AADC_AudiTT/camera_front/image_raw", Image, self.image_callback, queue_size=1, buff_size=65536*32)
        self.aer_pub = rospy.Publisher("/dvs/events", Image)

        rospy.loginfo("Waiting for image topics...")

    def image_callback(self, ros_image):
        print 'jaja mach!'
        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
            frame = self.bridge.imgmsg_to_cv2(ros_image)
        except CvBridgeError, e:
            print e

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayResized = cv2.resize(gray, (DVS_WIDTH, DVS_HEIGHT))

        img = cv2.resize(gray, (60, 30), interpolation=cv2.INTER_AREA)

        cv2.imshow('graublau', img)
        cv2.waitKey()
        image_message = self.bridge.cv2_to_imgmsg(gray, encoding="passthrough")
        self.aer_pub.publish(image_message)


    def process_image(self, diffImage, polarity):
        threshold = 20
        mask = cv2.threshold(diffImage, threshold, 255, cv2.THRESH_BINARY)
        event = cv2.findNonZero(mask[1])

        

        #if event is not None:
        #    aerEvents = [ Event(x=e[0][0],
        #                        y=e[0][1],
        #                        ts=rospy.Time.now(),
        #                        polarity=polarity) for e in event ]
        #else:
        #    aerEvents = []

        return event #aerEvents

    def cleanup(self):
        print "Shutting down vision node."


def main(args):
    try:
        frameToAERConverter()
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down vision node."

if __name__ == '__main__':
    main(sys.argv)
