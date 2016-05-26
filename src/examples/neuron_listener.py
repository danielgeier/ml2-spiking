#!/usr/bin/env python
# PKG = 'dvs_simulation'
from __future__ import division

import roslib  # ; roslib.load_manifest(PKG)

import rospy
import sys
import cv2
import std_msgs
# from dvs_msgs.msg import Event
# from dvs_msgs.msg import EventArray
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image, CameraInfo
from rospy.numpy_msg import numpy_msg
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import pyNN
from pyNN.nest import Population, SpikeSourcePoisson, SpikeSourceArray, AllToAllConnector, run, setup, IF_curr_alpha, \
    end
from pyNN.nest.projections import Projection

import matplotlib.pyplot as plt


class NeuronListener:
    def __init__(self):
        self.node_name = "neuron_listener"
        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.cleanup)

        self.bridge = CvBridge()


        self.image_sub = rospy.Subscriber("/dvs/events", Image, self.process_image)
        self.aer_pub = rospy.Publisher("/AADC_AudiTT/carUpdate", Vector3)
        #  expoisson.all_cells_source.


    def process_image(self, image):


        #print 'Bild ist da!'
        frame = self.bridge.imgmsg_to_cv2(image)
        # Set the image_data to pop
        cv2.imshow('graublau', frame)


        setup(timestep=0.1)

        # Orginal: NE = 960 * 1280
        NE = 30 * 100 / 2.0

      #  self.pop_in = Population((50, 100), SpikeSourcePoisson, {'rate': np.zeros(NE)})
        # self.pop_in_l = Population((50, 50), SpikeSourcePoisson, {'rate': np.zeros(NE)})
        # self.pop_in_r = Population((50, 50), SpikeSourcePoisson, {'rate': np.zeros(NE)})
        self.pop_in_l = Population((30, 50), IF_curr_alpha, {'i_offset': np.zeros(NE)})
        self.pop_in_r = Population((30, 50), IF_curr_alpha, {'i_offset': np.zeros(NE)})

        self.pop_out_l = Population(1, IF_curr_alpha, {'tau_refrac': 1, 'v_thresh' : -50})
        self.pop_out_r = Population(1, IF_curr_alpha, {'tau_refrac': 1, 'v_thresh' : -50})

        projection_l = Projection(self.pop_in_l, self.pop_out_l, AllToAllConnector())
        projection_r = Projection(self.pop_in_r, self.pop_out_r, AllToAllConnector())
        projection_l.setWeights(1.0)
        projection_r.setWeights(1.0)

        #frame_l = frame[0:50, 0:50]
        #frame_r = frame[0:50, 50:100]

        frame_l = frame[20:50, 0:50]
        frame_r = frame[20:50, 50:100]

        #cv2.imshow('lala', frame_l)
        #cv2.waitKey()
        #cv2.imshow('lala2', frame_r)
        #cv2.waitKey()

        self.pop_in_l.set(i_offset=frame_l.astype(float).flatten())
        self.pop_in_r.set(i_offset=frame_r.astype(float).flatten())

        self.pop_in_l.record('spikes')
        self.pop_in_r.record('spikes')
        self.pop_out_l.record('spikes')
        self.pop_out_r.record('spikes')

        tstop = 100.0
        run(tstop)

        # self.pop_out.printSpikes('out_spikes.h5')
        # self.pop_in.printSpikes('in_spikes.h5')

        #spikes_in = self.pop_in.get_data()
        data_out_l = self.pop_out_l.get_data()
        data_out_r = self.pop_out_r.get_data()

        end()

        num_spikes_l = data_out_l.segments[0].spiketrains[0].size
        num_spikes_r = data_out_r.segments[0].spiketrains[0].size

        # num_spikes_l = float(num_spikes_l)
        # num_spikes_r = float(num_spikes_r)

        num_spikes_diff = float(num_spikes_l) -float(num_spikes_r)
        print num_spikes_l
        print num_spikes_r
        #print num_spikes_diff


       # if (num_spikes_diff > num_spikes_r) :
        self.aer_pub.publish(Vector3(0.1, 0.0, num_spikes_diff)) # nach rechts lenken
       # else :
         #   self.aer_pub.publish(Vector3(1.0, 0.0, num_spikes_r))
      #links lenken




        #self.aer_pub.publish(Vector3(10.0, 0.0, num_spikes))

        # for seg in self.pop_in_l.get_data().segments:
        #     print seg
        #     for st in seg.spiketrains:
        #         print st



        # for seg in data_out.segments:
        #     print "SEGMENTS"
        #     print seg
        #     for asig in seg.analogsignals:
        #         print asig
        #     for st in seg.spiketrains:
        #         print st

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
