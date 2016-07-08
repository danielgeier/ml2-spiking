from __future__ import division

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Vector3
from pyNN import nest
from pyNN.nest import Population, AllToAllConnector, FromListConnector, IF_curr_alpha
from pyNN.nest.projections import Projection
from pyNN.random import RandomDistribution
from sensor_msgs.msg import Image

NUM_MIDDLE_LEARNING_LAYER = 5
NUM_INOUT_LEARNING_LAYER = 4

class SpikingNetworkNode:
    """Get retina images and store them. Publish to Gazebo."""

    def __init__(self):
        self.node_name = 'spiking_neuralnet'
        rospy.init_node(self.node_name, disable_signals=True)
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber('/spiky/retina_image', Image, self.save_frame)
        self.pub = rospy.Publisher('/AADC_AudiTT/carUpdate', Vector3, queue_size=1)
        self.last_frame = None
        # Make sure we get at least one frame
        rospy.wait_for_message('/spiky/retina_image', Image)
        # Make sure it is grayscale
        assert len(self.last_frame.shape) == 2

    def save_frame(self, ros_image):
        self.last_frame = self.bridge.imgmsg_to_cv2(ros_image)

    def publish(self, gas, brake, steering_angle):
        self.pub.publish(gas, brake, steering_angle)


class SpikingNetwork:
    def __init__(self, width, height):
        nest.setup(timestep=0.1)

        num_neurons = width * height

        self.sum_spikes_l = 0
        self.sum_spikes_r = 0

        self.pop_in_l = Population(num_neurons // 2, IF_curr_alpha, {'i_offset': np.zeros(num_neurons // 2)})
        self.pop_in_r = Population(num_neurons // 2, IF_curr_alpha, {'i_offset': np.zeros(num_neurons // 2)})

        # layer 2 links
        self.pop_in_l2 = Population(4, IF_curr_alpha, {'i_offset': np.zeros(4), 'v_thresh': 100})
        conn_list_l = []
        for neuron in self.pop_in_l:
            neuron = neuron - self.pop_in_l.first_id
            if (neuron % 50) <= 25 and neuron <= 1250:
                conn_list_l.append((neuron, 0, 1.0, 0.1))

            if (neuron % 50) > 25 and neuron <= 1250:
                conn_list_l.append((neuron, 1, 1.0, 0.1))

            if (neuron % 50) <= 25 and neuron > 1250:
                conn_list_l.append((neuron, 2, 1.0, 0.1))

            if (neuron % 50) > 25 and neuron > 1250:
                conn_list_l.append((neuron, 3, 1.0, 0.1))

        # layer2 rechts
        self.pop_in_r2 = Population(4, IF_curr_alpha, {'i_offset': np.zeros(4), 'v_thresh': 100})
        conn_list_r = []
        for neuron in self.pop_in_r:
            neuron = neuron - self.pop_in_r.first_id
            if (neuron % 50) <= 25 and neuron <= 1250:
                conn_list_r.append((neuron, 0, 1.0, 0.1))

            if (neuron % 50) > 25 and neuron <= 1250:
                conn_list_r.append((neuron, 1, 1.0, 0.1))

            if (neuron % 50) <= 25 and neuron > 1250:
                conn_list_r.append((neuron, 2, 1.0, 0.1))

            if (neuron % 50) > 25 and neuron > 1250:
                conn_list_r.append((neuron, 3, 1.0, 0.1))

        # Layer 3 output
        self.pop_out_l = Population(1, IF_curr_alpha, {'tau_refrac': 0.1, 'v_thresh': -50.})
        self.pop_out_r = Population(1, IF_curr_alpha, {'tau_refrac': 0.1, 'v_thresh': -50.})

        # Connections

        self.projection_layer2_l = Projection(self.pop_in_l, self.pop_in_l2, FromListConnector(conn_list_l))
        self.projection_layer2_r = Projection(self.pop_in_r, self.pop_in_r2, FromListConnector(conn_list_r))

        self.projection_layer2_l.setWeights(1.0)
        self.projection_layer2_r.setWeights(1.0)

        self.projection_out_l = Projection(self.pop_in_l2, self.pop_out_l, AllToAllConnector())
        self.projection_out_r = Projection(self.pop_in_r2, self.pop_out_r, AllToAllConnector())

        self.projection_out_l.setWeights(1.0)
        self.projection_out_r.setWeights(1.0)

        self.spikedetector_left = nest.nest.Create('spike_detector')
        self.spikedetector_right = nest.nest.Create('spike_detector')
        nest.nest.Connect(self.pop_out_l[0], self.spikedetector_left[0])
        nest.nest.Connect(self.pop_out_r[0], self.spikedetector_right[0])

        # net2 for Learning

        self.pop_learning_mid = Population(NUM_MIDDLE_LEARNING_LAYER, IF_curr_alpha, {'i_offset': np.zeros(NUM_MIDDLE_LEARNING_LAYER)})
        self.pop_learning_out = Population(2, IF_curr_alpha, {'i_offset': np.zeros(2)})

        self.projection_in_links = Projection(self.pop_learning_mid, self.pop_out_l, AllToAllConnector())
        self.projection_in_rechts = Projection(self.pop_learning_mid, self.pop_out_r, AllToAllConnector())
        self.projection_learning_out = Projection(self.pop_learning_out, self.pop_learning_mid, AllToAllConnector())


        self.spikedetector_l_out = np.ndarray(2,dtype='int32')
        for i in range(2):
            self.spikedetector_l_out[i] = nest.nest.Create('spike_detector')[0]
            nest.nest.Connect(self.pop_learning_out[i], self.spikedetector_l_out[i])

        self.spikedetector_l_mid = np.ndarray(NUM_MIDDLE_LEARNING_LAYER,dtype='int32')
        for i in range(NUM_MIDDLE_LEARNING_LAYER):
            self.spikedetector_l_mid[i] = nest.nest.Create('spike_detector')[0]
            nest.nest.Connect(self.pop_learning_mid[i], self.spikedetector_l_mid[i])


        vthresh_distr = RandomDistribution('uniform', [0.1, 1])
        print vthresh_distr

        self.projection_in_links.setWeights(vthresh_distr)
        self.projection_in_rechts.setWeights(vthresh_distr)
        self.projection_learning_out.setWeights(vthresh_distr)

        print self.projection_in_links.getWeights()

        elig = np.zeros((NUM_MIDDLE_LEARNING_LAYER,NUM_INOUT_LEARNING_LAYER))

        count = 0
        for neuron in self.pop_learning_mid:
            source_con =  nest.nest.GetConnections(source=[neuron])
            target_con =  nest.nest.GetConnections(target=[neuron])

            elig[count][0] = nest.nest.GetStatus(source_con, 'weight')[1]   #1 bzw 2 wegen verbindung zum spikedetector
            elig[count][1] = nest.nest.GetStatus(source_con, 'weight')[2]
            elig[count][2] = nest.nest.GetStatus(target_con, 'weight')[0]
            elig[count][3] = nest.nest.GetStatus(target_con, 'weight')[1]
            count = count+1

        print elig


    def inject(self, frame):
        frame_l = frame[0:50, 0:50]
        frame_r = frame[0:50, 50:100]

        self.pop_in_l.set(i_offset=frame_l.astype(float).flatten())
        self.pop_in_r.set(i_offset=frame_r.astype(float).flatten())

        tstop = 20.0 #
        nest.run(tstop)
        nest.end()


        spikes_array = np.ndarray((NUM_MIDDLE_LEARNING_LAYER+NUM_INOUT_LEARNING_LAYER),dtype='int32')
        spikes_array[0] = 1 if (nest.nest.GetStatus(self.spikedetector_left, 'n_events')[0] >= 1) else 0
        spikes_array[1] = 1 if (nest.nest.GetStatus(self.spikedetector_right, 'n_events')[0] >= 1) else 0
        spikes_array[2] = 1 if (nest.nest.GetStatus([self.spikedetector_l_mid[0]], 'n_events')[0] >= 1) else 0
        spikes_array[3] = 1 if (nest.nest.GetStatus([self.spikedetector_l_mid[1]], 'n_events')[0] >= 1) else 0
        spikes_array[4] = 1 if (nest.nest.GetStatus([self.spikedetector_l_mid[2]], 'n_events')[0] >= 1) else 0
        spikes_array[5] = 1 if (nest.nest.GetStatus([self.spikedetector_l_mid[3]], 'n_events')[0] >= 1) else 0
        spikes_array[6] = 1 if (nest.nest.GetStatus([self.spikedetector_l_mid[4]], 'n_events')[0] >= 1) else 0
        spikes_array[7] = 1 if (nest.nest.GetStatus([self.spikedetector_l_out[0]], 'n_events')[0] >= 1) else 0
        spikes_array[8] = 1 if (nest.nest.GetStatus([self.spikedetector_l_out[1]], 'n_events')[0] >= 1) else 0

        print spikes_array

        num_spikes_l = nest.nest.GetStatus(self.spikedetector_left, "n_events")[0]
        num_spikes_r = nest.nest.GetStatus(self.spikedetector_right, "n_events")[0]
        nest.nest.SetStatus(self.spikedetector_left, "n_events", 0)
        nest.nest.SetStatus(self.spikedetector_right, "n_events", 0)

        num_spikes_diff = num_spikes_l - num_spikes_r
        # TODO ensure -1 <= angle <= 1
        angle = num_spikes_diff / 15
        brake = 0  # np.exp(abs(angle)) - 1
        gas = 1 / (abs(angle) + 1.5)
        print 'l {:3d} | r {:3d} | diff {:3d} | gas {:2.2f} | brake {:2.2f} | steer {:2.2f}'.format(
            num_spikes_l,
            num_spikes_r,
            num_spikes_diff,
            gas,
            brake,
            angle)

        return gas, brake, angle


def main():
    node = SpikingNetworkNode()
    w, h = node.last_frame.shape
    net = SpikingNetwork(w, h)
    window = cv2.namedWindow('Cam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, w * 4, h * 4)

    try:
        while True:
            frame = node.last_frame
            gas, brake, angle = net.inject(frame)

            frame2 = frame.copy()
            frame2.T[50] = 255 - frame2.T[50]
            cv2.imshow('Cam', frame2)
            cv2.waitKey(1)

            node.publish(gas, brake, angle)
    except Exception, e:
        print e


if __name__ == '__main__':
    main()
