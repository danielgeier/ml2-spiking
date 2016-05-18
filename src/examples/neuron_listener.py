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
import pyNN
from pyNN.nest import Population, SpikeSourcePoisson, SpikeSourceArray


class NeuronListener:
    def __init__(self):
        self.node_name = "neuron_listener"
        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.cleanup)

        self.bridge = CvBridge()
        NE = 960 * 1280

        self.pop = create(NE, SpikeSourceArray, {'spike_times': [0 for i in range (1,1,NE)]})
        self.image_sub = rospy.Subscriber("/dvs/events", Image, self.process_image)
      #  expoisson.all_cells_source.




    def process_image(self, image):

        print 'Bild ist da!'
        frame = self.bridge.imgmsg_to_cv2(image)
        self.pop.tset("spike_times", image.data)
        cv2.imshow('graublau', frame)
        cv2.waitKey()

    def cleanup(self):
        print "Shutting down vision node."


def main(args):
    try:
        NeuronListener()
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down vision node."

if __name__ == '__main__':
    main(sys.argv)

