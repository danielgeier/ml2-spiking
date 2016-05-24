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
from pyNN.nest import Population, SpikeSourcePoisson, SpikeSourceArray, AllToAllConnector, run, setup, IF_curr_alpha
from pyNN.nest.projections import Projection

class NeuronListener:
    def __init__(self):
        self.node_name = "neuron_listener"
        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.cleanup)

        self.bridge = CvBridge()

        setup(timestep=0.1)

        # Orginal: NE = 960 * 1280
        NE=60*30

        self.pop_in = Population(NE, SpikeSourcePoisson, {'rate': np.zeros(NE)})
        self.pop_out = Population(1, IF_curr_alpha, {})

        projection = Projection(self.pop_in, self.pop_out, AllToAllConnector())
        projection.setWeights(1.0)

        self.pop_in.record('spikes')
        self.pop_out.record('spikes')

        tstop = 1000.0
        run(tstop)

        self.pop_in.write_data("simpleNetwork_output.pkl", 'spikes')
        self.pop_out.write_data("simpleNetwork_input.pkl", 'spikes')


        self.image_sub = rospy.Subscriber("/dvs/events", Image, self.process_image)
      #  expoisson.all_cells_source.




    def process_image(self, image):

        print 'Bild ist da!'
        frame = self.bridge.imgmsg_to_cv2(image)
        # Set the image_data to pop
        self.pop_in.set(rate=frame)
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

