#!/usr/bin/env python
# PKG = 'dvs_simulation'
from __future__ import division
from Queue import Queue
from time import sleep


import rospy
import sys
import cv2
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from pyNN.nest import Population, SpikeSourcePoisson, SpikeSourceArray, AllToAllConnector, run, setup, IF_curr_alpha, \
    end, reset
from pyNN.nest.projections import Projection

import matplotlib.pyplot as plt


class NeuronListener:
    def __init__(self):
        self.node_name = "neuron_listener"
        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.cleanup)

        self.frame_queue = Queue(maxsize=1)

        self.bridge = CvBridge()
        self.sum_spikes_l = 0
        self.sum_spikes_r = 0


        self.image_sub = rospy.Subscriber("/dvs/events", Image, self.process_image, queue_size=1)
        self.aer_pub = rospy.Publisher("/AADC_AudiTT/carUpdate", Vector3)
        #  expoisson.all_cells_source.

        # Orginal: NE = 960 * 1280
        NE = 30 * 100 / 2.0

        setup(timestep=0.1)

        self.pop_in_l = Population((30, 50), IF_curr_alpha, {'i_offset': np.zeros(NE)})
        self.pop_in_r = Population((30, 50), IF_curr_alpha, {'i_offset': np.zeros(NE)})

        self.pop_out_l = Population(1, IF_curr_alpha, {'tau_refrac': 1, 'v_thresh' : -50})
        self.pop_out_r = Population(1, IF_curr_alpha, {'tau_refrac': 1, 'v_thresh' : -50})

        projection_l = Projection(self.pop_in_l, self.pop_out_l, AllToAllConnector())
        projection_r = Projection(self.pop_in_r, self.pop_out_r, AllToAllConnector())
        projection_l.setWeights(1.0)
        projection_r.setWeights(1.0)

        self.pop_in_l.record('spikes')
        self.pop_in_r.record('spikes')
        self.pop_out_l.record('spikes')
        self.pop_out_r.record('spikes')


    def process_image(self, image):
        #print 'Bild ist da!'
        frame = self.bridge.imgmsg_to_cv2(image)
        # Set the image_data to pop
        cv2.imshow('graublau', frame)

        if self.frame_queue.full():
            self.frame_queue.get()

        self.frame_queue.put(frame)



    def inject(self, frame):
        frame_l = frame[20:50, 0:50]
        frame_r = frame[20:50, 50:100]

        cv2.imshow('lala', frame_l)
        cv2.waitKey(1)
        cv2.imshow('lala2', frame_r)
        cv2.waitKey(1)

        self.pop_in_l.set(i_offset=frame_l.astype(float).flatten())
        self.pop_in_r.set(i_offset=frame_r.astype(float).flatten())

        tstop = 100.0
        run(tstop)

        #spikes_in = self.pop_in.get_data()
        data_out_l = self.pop_out_l.get_data()
        data_out_r = self.pop_out_r.get_data()

        end()

        num_spikes_l = data_out_l.segments[0].spiketrains[0].size - self.sum_spikes_l
        num_spikes_r = data_out_r.segments[0].spiketrains[0].size - self.sum_spikes_r

        self.sum_spikes_l += num_spikes_l
        self.sum_spikes_r += num_spikes_r

        num_spikes_diff = float(num_spikes_l) -float(num_spikes_r)
        print num_spikes_l, num_spikes_r
        #print num_spikes_diff

        return Vector3(0.3, 0.0, num_spikes_diff) # nach rechts lenken

    def cleanup(self):
        print "Shutting down vision node."

    def next_frame(self):
        return self.frame_queue.get()


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
    net = NeuronListener()

    try:
        while True:
            sleep(0.001)
            frame = net.next_frame()
            out = net.inject(frame)
            net.aer_pub.publish(out)
            print out

    except KeyboardInterrupt:
        print "Shutting down vision node."


if __name__ == '__main__':
    main(sys.argv)
