#!/usr/bin/env python
# PKG = 'dvs_simulation'
import roslib  # ; roslib.load_manifest(PKG)

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

import matplotlib.pyplot as plt


class NeuronListener:
    def __init__(self):
        self.node_name = "neuron_listener"
        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.cleanup)

        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber("/dvs/events", Image, self.process_image)
        #  expoisson.all_cells_source.

    def process_image(self, image):

        setup(timestep=0.1)

        # Orginal: NE = 960 * 1280
        NE = 50 * 100

        self.pop_in = Population((50, 100), SpikeSourcePoisson, {'rate': np.zeros(NE)})
        self.pop_out = Population(1, IF_curr_alpha, {'tau_refrac': 5 })

        projection = Projection(self.pop_in, self.pop_out, AllToAllConnector())
        projection.setWeights(1.0)

        print 'Bild ist da!'
        frame = self.bridge.imgmsg_to_cv2(image)
        # Set the image_data to pop
        self.pop_in.set(rate=frame.astype(float).flatten())
        cv2.imshow('graublau', frame)

        self.pop_in.record('spikes')
        self.pop_out.record('spikes')

        tstop = 100.0
        run(tstop)

        # self.pop_out.printSpikes('out_spikes.h5')
        # self.pop_in.printSpikes('in_spikes.h5')

        spikes_in = self.pop_in.get_data()
        data_out = self.pop_out.get_data()

        # for seg in spikes_in.segments:
        #     print seg
        #     for asig in seg.analogsignals:
        #         print asig
        #     for st in seg.spiketrains:
        #         print st



        for seg in data_out.segments:
            print seg
            for asig in seg.analogsignals:
                print asig
            for st in seg.spiketrains:
                print st

                    # n_panels = sum(a.shape[1] for a in data_out.segments[0].analogsignalarrays) + 2
                    # plt.subplot(n_panels, 1, 1)
                    # plot_spiketrains(spikes_in.segments[0])
                    # plt.subplot(n_panels, 1, 2)
                    # plot_spiketrains(data_out.segments[0])
                    # panel = 3
                    # for array in data_out.segments[0].analogsignalarrays:
                    #     for i in range(array.shape[1]):
                    #         plt.subplot(n_panels, 1, panel)
                    #         plot_signal(array, i, colour='bg'[panel % 2])
                    #         panel += 1
                    # plt.xlabel("time (%s)" % array.times.units._dimensionality.string)
                    # plt.setp(plt.gca().get_xticklabels(), visible=True)
                    #
                    #
                    # plt.show()

    def cleanup(self):
        print "Shutting down vision node."


def plot_spiketrains(segment):
    for spiketrain in segment.spiketrains:
        y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
        plt.plot(spiketrain, y, '.')
        plt.ylabel(segment.name)
        plt.setp(plt.gca().get_xticklabels(), visible=False)


def plot_signal(signal, index, colour='b'):
    label = "Neuron %d" % signal.annotations['source_ids'][index]
    plt.plot(signal.times, signal[:, index], colour, label=label)
    plt.ylabel("%s (%s)" % (signal.name, signal.units._dimensionality.string))
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.legend()


def main(args):
    try:
        NeuronListener()
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down vision node."


if __name__ == '__main__':
    main(sys.argv)
