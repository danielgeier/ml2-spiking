from __future__ import division

from abc import ABCMeta, abstractmethod, abstractproperty

import cv2
import numpy as np
import rospy
import argparse

from cv_bridge import CvBridge
from geometry_msgs.msg import Vector3
from pyNN import nest
from pyNN.nest import Population, AllToAllConnector, FromListConnector, IF_curr_alpha
from pyNN.nest.projections import Projection
from pyNN.random import RandomDistribution
from sensor_msgs.msg import Image
from snn_plotter.proxy import PlotterProxy
from vehicle_control.srv import *

import os
import time
import datetime
import logging.handlers as handlers
import logging
import threading
import Tkinter as Tk

from std_msgs.msg import Bool, Float64MultiArray

# Logging/Dumping locations
DUMPS_DIR = '../dumps'
WEIGHTS_DUMPS_DIR = os.path.join(DUMPS_DIR, 'weights')

NUM_IN_LEARNING_LAYER = 2
NUM_MIDDLE_LEARNING_LAYER = 50
NUM_OUT_LEARNING_LAYER = 2
NUM_RL_NEURONS = NUM_MIDDLE_LEARNING_LAYER + NUM_OUT_LEARNING_LAYER
IN_WEIGHTS = 1.0 # Feature Layer to Mid Layer (Learning)
TIME_STEP = 0.1 # ?

# Parameters for reinforcement learning
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.5
NUM_TRACE_STEPS = 10 # TODO better name
TAU = 2.  # ?
BETA_SIGMA = 0.3  # ?
SIGMA = 0.001


class LaneletInformation:
    def __init__(self, data):
        self.distance = data[0]
        self.lanelet_angle = data[2]

        self.is_on_lane = bool(data[1])
        self.is_left = bool(data[3])
        self.is_right = not self.is_left

    def __str__(self):
        s = """
        Is on Lane: %s
        Is Left: %s
        Is Right: %s
        Distance: %.2f
        Angle of Vehicle to Lanelet (rad): %.2f
        """ % (self.is_on_lane, self.is_left, self.is_right, self.distance, self.lanelet_angle)

        return s


class Learner:
    __metaclass__ = ABCMeta

    def __init__(self, network, world):
        self._network = network
        self._world = world

    @property
    def network(self):
        return self._network

    @property
    def world(self):
        return self._world

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class ReinforcementLearner(Learner):
    """ See 'A reinforcement learning algorithm for spiking neural networks', Razvan V. Florian, 2005, CN, Romania"""

    def __init__(self, network, world, beta_sigma, sigma, tau, steps_to_trace, output_neurons_count, discount_factor,
                 learning_rate, learningrate_delta=0.01):

        super(ReinforcementLearner, self).__init__(network, world)
        self._beta_sigma = beta_sigma
        self._sigma = sigma
        self._tau = tau
        self._steps_to_trace = steps_to_trace
        self._output_neurons_count = output_neurons_count
        self._learning_rate = learning_rate
        self._learningrate_delta = learningrate_delta
        self._discount_factor = discount_factor

        self._connections = nest.nest.GetConnections(target=self.network.postsynaptic_learning_neurons)
        self._eligibility_trace = np.zeros(len(self._connections))

        # Initialize spike dictionary for each post
        self._spikes = {}
        self._neurons = set([])
        self._weights = self.network.get_weights()

        # Extract all relevant neurons, i.e. all neurons connected with postsynaptic learning neurons and
        # postsynaptic neurons themselves
        for connection in self._connections:
            self._neurons.add(connection[0])
            self._neurons.add(connection[1])

        for neuron in self._neurons:
            # we trace one more step, since we need the steps_to_trace last steps and the current step
            self._spikes[neuron] = np.zeros(steps_to_trace + 1)

    def learn(self):
        self._update_spikes()
        self._update_weights()

        print "z: ", self._eligibility_trace
        print "w: ", self._weights

    def _update_spikes(self):
        # TODO: This might be slower than previous implementation ... it would be better to sort
        # the neurons and use the permutation for a matrix instead of looking up stuff in a dictionary
        events_spike_detectors = nest.nest.GetStatus([self.network.detectors[x] for x in self._neurons], keys='events')
        senders = set([])
        for event in events_spike_detectors:
            if len(event['senders']) > 0:
                sender = event['senders'][0]
                senders.add(sender)
                self._spikes[sender][0] = 1 #len(event['senders'])

        idle_neurons = self._neurons.difference(senders)
        for neuron in idle_neurons:
            self._spikes[neuron][0] = 0

        # Shift spike events to the left with rollover (1 0 2) -> (0 2 1)
        for key in self._spikes.keys():
            self._spikes[key] = np.roll(self._spikes[key], -1)

    def reset(self):
        pass

    def _update_weights(self):
        # get reward from world
        reward = self.world.calculate_reward(self.network.decode_actions())

        # current weights
        weights = np.array(nest.nest.GetStatus(self._connections, 'weight'))

        # exp(-(k-1)*delta_t/tau)
        decay = np.exp(-np.arange( self._steps_to_trace) * self.network.timestep / self._tau )
        i = 0
        for connection in self._connections:
            postsyn_neuron = connection[1]
            presyn_neuron = connection[0]

            zeta = np.sum(self._spikes[presyn_neuron][:-1] * decay)
            if self._spikes[postsyn_neuron][-1] > 0:
                zeta *= self._beta_sigma

            else:
                zeta *= -self._beta_sigma * self._sigma/(1.0 - self._sigma)

            self._eligibility_trace[i] = self._discount_factor*self._eligibility_trace[i] + zeta
            weights[i] += self._learning_rate * reward * self._eligibility_trace[i]

            i += 1

        nest.nest.SetStatus(self._connections, [{'weight': w} for w in weights])
        self._weights = weights


class BaseNetwork:
    """ Defines a basic network with an input layer """
    __metaclass__ = ABCMeta

    def __init__(self, timestep, simduration, width, height):

        self._timestep = timestep
        self._simduration = simduration
        self._detectors = {}
        self._weights = None
        self._width = width
        self._height = height

        # Populations
        self.pop_input_image_r = None
        self.pop_input_image_l = None
        self.pop_encoded_image_l = None
        self.pop_encoded_image_r = None

        self.spikedetector_enc_image_l = None
        self.spikedetector_enc_image_r = None

        self._build_input_layer()
        self._create_spike_detectors()

    def _build_input_layer(self):
        num_neurons = self._width * self._height
        # Layer 1 (Input Image)
        self.pop_input_image_r = Population(num_neurons // 2, IF_curr_alpha, {'i_offset': np.zeros(num_neurons // 2)})
        self.pop_input_image_l = Population(num_neurons // 2, IF_curr_alpha, {'i_offset': np.zeros(num_neurons // 2)})

        # Two 4 x 4 grids, for left and right respectively
        # Layer 2: 4 x 4 Grid left
        self.pop_in_l2 = Population(4, IF_curr_alpha, {'i_offset': np.zeros(4), 'v_thresh': 100})
        conn_list_l = []
        for neuron in self.pop_input_image_l:
            neuron = neuron - self.pop_input_image_l.first_id
            if (neuron % 50) <= 25 and neuron <= 1250:  # oben links
                conn_list_l.append((neuron, 0, 1.0, 0.1))

            if (neuron % 50) > 25 and neuron <= 1250:  # oben rechts
                conn_list_l.append((neuron, 1, 1.0, 0.1))

            if (neuron % 50) <= 25 and neuron > 1250:  # unten links
                conn_list_l.append((neuron, 2, 1.0, 0.1))

            if (neuron % 50) > 25 and neuron > 1250:  # unten rechts   DOPPELT gewichtet
                conn_list_l.append((neuron, 3, 1.0, 0.1))

        # Layer 2: 4 x 4 Grid right
        self.pop_in_r2 = Population(4, IF_curr_alpha, {'i_offset': np.zeros(4), 'v_thresh': 100})
        conn_list_r = []
        for neuron in self.pop_input_image_r:
            neuron = neuron - self.pop_input_image_r.first_id
            if (neuron % 50) <= 25 and neuron <= 1250:  # oben links
                conn_list_r.append((neuron, 0, 1.0, 0.1))

            if (neuron % 50) > 25 and neuron <= 1250:  # oben recht
                conn_list_r.append((neuron, 1, 1.0, 0.1))

            if (neuron % 50) <= 25 and neuron > 1250:  # unten links DOPPELT gewichtet
                conn_list_r.append((neuron, 2, 1.0, 0.1))

            if (neuron % 50) > 25 and neuron > 1250:  # unten rechts
                conn_list_r.append((neuron, 3, 1.0, 0.1))

        # Layer 3 encoded ( The two 4x4 Grids are connected with one neuron each)
        self.pop_encoded_image_l = Population(1, IF_curr_alpha, {'tau_refrac': 0.1, 'v_thresh': -50.})
        self.pop_encoded_image_r = Population(1, IF_curr_alpha, {'tau_refrac': 0.1, 'v_thresh': -50.})

        # Connections: Layer 1 (Input Image) --> Layer 2 (4 x 4 Grids)
        self.projection_layer2_l = Projection(self.pop_input_image_l, self.pop_in_l2, FromListConnector(conn_list_l))
        self.projection_layer2_r = Projection(self.pop_input_image_r, self.pop_in_r2, FromListConnector(conn_list_r))
        self.projection_layer2_l.setWeights(1.0)
        self.projection_layer2_r.setWeights(1.0)

        # Connections: Layer 2 (4 x 4 Grids) --> Layer 3 (Encoded Image)
        self.projection_out_l = Projection(self.pop_in_l2, self.pop_encoded_image_l, AllToAllConnector())
        self.projection_out_r = Projection(self.pop_in_r2, self.pop_encoded_image_r, AllToAllConnector())
        self.projection_out_l.setWeights(1.0)
        self.projection_out_r.setWeights(1.0)

    def populate_plotter(self, plotter):
        plotter.add_spike_train_plot(self.pop_encoded_image_l, label='Encoded Image Left')
        plotter.add_spike_train_plot(self.pop_encoded_image_r, label='Encoded Image Right')

    def _create_spike_detectors(self):
        self.spikedetector_enc_image_l = nest.nest.Create('spike_detector', params={'withgid': True, 'withtime': True})[0]
        self.spikedetector_enc_image_r = nest.nest.Create('spike_detector', params={'withgid': True, 'withtime': True})[0]

        nest.nest.Connect(self.pop_encoded_image_l[0], self.spikedetector_enc_image_l)
        nest.nest.Connect(self.pop_encoded_image_r[0], self.spikedetector_enc_image_r)

        self.detectors[self.pop_encoded_image_l[0]] = self.spikedetector_enc_image_l
        self.detectors[self.pop_encoded_image_r[0]] = self.spikedetector_enc_image_r

    @property
    def encoded_image_pops(self):
        return {'left': self.pop_encoded_image_l, 'right': self.pop_encoded_image_r}

    @property
    def plotter(self):
        return self._plotter

    @abstractproperty
    def output_pop(self):
        pass

    @property
    def timestep(self):
        return self._timestep

    @property
    def simduration(self):
        return self._simduration

    @property
    def detectors(self):
        return self._detectors

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def set_weights(self, value):
        pass

    @abstractmethod
    def reset_weights(self):
        pass

    @abstractmethod
    def decode_actions(self):
        pass

    @abstractmethod
    def postsynaptic_learning_neurons(self):
        pass

    def inject_frame(self, frame, lanelet_information):
        frame_l = frame[0:50, 0:50]
        frame_r = frame[0:50, 50:100]

        self.pop_input_image_l.set(i_offset=frame_l.astype(float).flatten())
        self.pop_input_image_r.set(i_offset=frame_r.astype(float).flatten())

        return 0, 0, 0

    def simulate(self):
        nest.nest.SetStatus(self.detectors.values(), 'n_events', 0)  # reset detectors before simulating the next step
        nest.run(self._simduration)
        nest.end()


class BraitenbergNetwork(BaseNetwork):
    def __init__(self, timestep, simduration, width, height):
        super(BraitenbergNetwork, self).__init__(timestep, simduration, width, height)

        self._output_pop = None
        self._left_connection = None
        self._right_connection = None
        self._postsynaptic_learning_neurons = []

        self._build_network()

    def _build_network(self):
        self._output_pop = Population(2, IF_curr_alpha, {'tau_refrac': 0.1, 'i_offset': np.zeros(2)})
        l = self.encoded_image_pops['left']
        r = self.encoded_image_pops['right']

        self._left_connection = Projection(l, self._output_pop, AllToAllConnector())
        self._right_connection = Projection(r, self._output_pop, AllToAllConnector())

        # Detectors
        spikedetector_left = nest.nest.Create('spike_detector', params={'withgid': True, 'withtime': True})[0]
        spikedetector_right = nest.nest.Create('spike_detector', params={'withgid': True, 'withtime': True})[0]

        nest.nest.Connect(self._output_pop[0], spikedetector_left)
        nest.nest.Connect(self._output_pop[1], spikedetector_right)

        self.detectors[self._output_pop[0]] = spikedetector_right
        self.detectors[self._output_pop[1]] = spikedetector_left

        # Here go the neurons whose presynaptic connections are
        # adapted by a learner. In this case, we only want to adapt
        # the presynaptic connections of the output neurons.
        for neuron in self._output_pop:
            self._postsynaptic_learning_neurons.append(neuron)

        self.reset_weights()

    @property
    def postsynaptic_learning_neurons(self):
        return self._postsynaptic_learning_neurons


    def populate_plotter(self, plotter):
        super(BraitenbergNetwork, self).populate_plotter(plotter)
        plotter.add_spike_train_plot(self._output_pop, 'Output L/R')

    @property
    def output_pop(self):
        return self._output_pop

    def reset_weights(self):
        weights = np.random.rand(2, 2)*1000
        self.set_weights(weights)

    def get_weights(self):
        target_left = nest.nest.GetConnections(target=[self._output_pop[0]])
        target_right = nest.nest.GetConnections(target=[self._output_pop[1]])

        weights_l = nest.nest.GetStatus(target_left, 'weight')
        weights_r = nest.nest.GetStatus(target_right, 'weight')

        return np.vstack((weights_l, weights_r))

    def set_weights(self, weights):
        self._left_connection.setWeights(weights[:, 0]/1000)
        self._right_connection.setWeights(weights[:, 1]/1000)

    def decode_actions(self):
        spikes = np.array(nest.nest.GetStatus([self.detectors[x] for x in self._output_pop], 'n_events'))

        # last two are output neurons
        num_spikes_l = spikes[0]
        num_spikes_r = spikes[1]

        num_spikes_diff = num_spikes_l - num_spikes_r
        angle = num_spikes_diff / 10  # minus = rechts
        brake = 0
        gas = 1 / (abs(angle) + 2.5)

        return {'gas': gas, 'brake': brake, 'steering_angle': angle}


class BaseWorld:
    __metaclass__ = ABCMeta
    
    def __init__(self):
        pass

    @abstractmethod
    def calculate_reward(self, actions):
        pass

    @abstractproperty
    def current_state(self):
        pass


class World(BaseWorld):
    def __init__(self,topic='/laneletInformation'):
        super(World, self).__init__()
        self.sub_lanelet = rospy.Subscriber(topic, Float64MultiArray, self._update_state,
                                            queue_size=1)
        self._state = None

    def calculate_reward(self, actions):
        state = self._state

        if state is None:
            return 0

        steering_angle = actions['steering_angle']
        is_on_lanelet = state.is_on_lane

        # distance from the center of the right lane
        distance = state.distance
        if is_on_lanelet:
            reward = (1 - distance)
        else:
            if (state.is_left and steering_angle < 0) or (state.is_right and steering_angle > 0):
                reward = 0.1
            else:
                reward = -10. * (distance + 1) * (abs(steering_angle) + 1)

        if reward < -20:
            reward = -20

        return reward*10

    @property
    def current_state(self):
        return self._state

    def _update_state(self, state):
        if len(state.data) > 0:
            self._state = LaneletInformation(state.data)


class DeepNetwork(BaseNetwork):
    def __init__(self, timestep, simduration):
        super(DeepNetwork, self).__init__(timestep, simduration)


class NetworkPlotter:
    def __init__(self, network, plot_steps=10):
        self._network = network
        self._plotter = PlotterProxy(network.simduration, plot_steps)
        network.populate_plotter(self._plotter)

    def update(self):
        self._plotter.update(nest.get_current_time())


class SpikingNetworkNode:
    """Get retina images and store them. Publish to Gazebo."""

    def __init__(self):
        # type: () -> SpikingNetworkNode
        # Most recent time step is in last array position
        # Neurons in row, spikes for current time step in column
        #self.spikes = np.zeros((NUM_IN_LEARNING_LAYER + NUM_RL_NEURONS, NUM_TRACE_STEPS),dtype='int32')
        self.node_name = 'spiking_neuralnet'
        self.is_set_back = None
        rospy.init_node(self.node_name, disable_signals=True)
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber('/spiky/retina_image', Image, self.save_frame, queue_size=1)
        self.sub_lanelet = rospy.Subscriber('/laneletInformation', Float64MultiArray, self.update_car_state, queue_size=1)
        self.sub_is_set_back = rospy.Subscriber('/AADC_AudiTT/isSetBack', Bool, self.set_param_back, queue_size=1)
        self.pub = rospy.Publisher('/AADC_AudiTT/carUpdate', Vector3, queue_size=1)
        self.last_frame = None
        self.last_distance = None
        self.lanelet_information = None
        # Make sure we get at least one frame
        rospy.wait_for_message('/spiky/retina_image', Image)
        # Make sure it is grayscale
        assert len(self.last_frame.shape) == 2

    def save_frame(self, ros_image):
        self.last_frame = self.bridge.imgmsg_to_cv2(ros_image)

    def publish(self, gas, brake, steering_angle):
        self.pub.publish(gas, brake, steering_angle)

    def update_car_state(self, lanelet_info):
        if len(lanelet_info.data) > 0:
            self.lanelet_information = LaneletInformation(lanelet_info.data)
            self.last_distance = lanelet_info.data[0]

    def set_param_back(self, isSetBack):
        self.is_set_back = isSetBack.data


class SpikingNetwork:
    def __init__(self, width, height, plot):
        # Most recent time step is in last array position
        # Neurons in row, spikes for current time step in column
        self.spikes = np.zeros((NUM_RL_NEURONS, NUM_TRACE_STEPS), dtype='int32')
        self.reward = 0
        self.last_distance = 0
        nest.setup(timestep=TIME_STEP)
        self.eligibility_trace = np.zeros((NUM_MIDDLE_LEARNING_LAYER,NUM_OUT_LEARNING_LAYER))
        self.weights = np.zeros((NUM_MIDDLE_LEARNING_LAYER, NUM_OUT_LEARNING_LAYER))
        self.learningrate_fakt = LEARNING_RATE

        # Flags
        self.plot = plot
        self.learn = True

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
            if (neuron % 50) <= 25 and neuron <= 1250: # oben links
                conn_list_l.append((neuron, 0, 1.0, 0.1))

            if (neuron % 50) > 25 and neuron <= 1250: # oben rechts
                conn_list_l.append((neuron, 1, 1.0, 0.1))

            if (neuron % 50) <= 25 and neuron > 1250: # unten links
                conn_list_l.append((neuron, 2, 1.0, 0.1))

            if (neuron % 50) > 25 and neuron > 1250: # unten rechts   DOPPELT gewichtet
                conn_list_l.append((neuron, 3, 1.0, 0.1))

        # layer2 rechts
        self.pop_in_r2 = Population(4, IF_curr_alpha, {'i_offset': np.zeros(4), 'v_thresh': 100})
        conn_list_r = []
        for neuron in self.pop_in_r:
            neuron = neuron - self.pop_in_r.first_id
            if (neuron % 50) <= 25 and neuron <= 1250: # oben links
                conn_list_r.append((neuron, 0, 1.0, 0.1))

            if (neuron % 50) > 25 and neuron <= 1250: # oben recht
                conn_list_r.append((neuron, 1, 1.0, 0.1))

            if (neuron % 50) <= 25 and neuron > 1250: #unten links DOPPELT gewichtet
                conn_list_r.append((neuron, 2, 1.0, 0.1))

            if (neuron % 50) > 25 and neuron > 1250: # unten rechts
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

        # bild mitte doppelt gewichtet
        # self.projection_out_l[3].setWeights(2.0)
        # self.projection_out_r[2].setWeights(2.0)
        self.projection_out_l[3]._set_weight(2.0)
        self.projection_out_r[2]._set_weight(2.0)

        # Create Detectors
        self.spikedetector_left = nest.nest.Create('spike_detector')[0]
        self.spikedetector_right = nest.nest.Create('spike_detector')[0]
        nest.nest.Connect(self.pop_out_l[0], self.spikedetector_left)
        nest.nest.Connect(self.pop_out_r[0], self.spikedetector_right)

        # net2 for Learning
        self.pop_learning_mid = Population(NUM_MIDDLE_LEARNING_LAYER, IF_curr_alpha, {'tau_refrac': 0.1, 'i_offset': np.zeros(NUM_MIDDLE_LEARNING_LAYER)})#'v_thresh' : -64})
        self.pop_learning_out = Population(2, IF_curr_alpha, {'tau_refrac': 0.1, 'i_offset': np.zeros(2)})

        self.spikedetector_l_out = np.ndarray(2,dtype='int32')
        for i in range(2):
            self.spikedetector_l_out[i] = nest.nest.Create('spike_detector')[0]
            nest.nest.Connect(self.pop_learning_out[i], self.spikedetector_l_out[i])

        self.spikedetector_l_mid = np.ndarray(NUM_MIDDLE_LEARNING_LAYER,dtype='int32')
        for i in range(NUM_MIDDLE_LEARNING_LAYER):
            self.spikedetector_l_mid[i] = nest.nest.Create('spike_detector')[0]
            nest.nest.Connect(self.pop_learning_mid[i], self.spikedetector_l_mid[i])

        #self.all_detectors = np.append([self.spikedetector_left, self.spikedetector_right], self.spikedetector_l_mid)
        #self.all_detectors = np.append(self.all_detectors, self.spikedetector_l_out)
        self.all_detectors = np.append(self.spikedetector_l_mid,self.spikedetector_l_out)
        self.all_detectors = list(self.all_detectors)

        # Connect Layers
        self.projection_in_links = Projection(self.pop_out_l, self.pop_learning_mid,  AllToAllConnector())
        self.projection_in_rechts = Projection(self.pop_out_r, self.pop_learning_mid, AllToAllConnector())
        self.projection_learning_out = Projection(self.pop_learning_mid, self.pop_learning_out, AllToAllConnector())

        vthresh_distr_1 = RandomDistribution('uniform', [0.1, 2])
        self.projection_in_links.setWeights(vthresh_distr_1)
        self.projection_in_rechts.setWeights(vthresh_distr_1)

        self.reset_weights()

        if self.plot:
            plot_steps = 10
            self.proxy = PlotterProxy(20., plot_steps)
            self.proxy.add_spike_train_plot(self.pop_out_l, label='Learning In_L')
            self.proxy.add_spike_train_plot(self.pop_out_r, label='Learning In_R')
            self.proxy.add_spike_train_plot(self.pop_learning_mid, label='Learning Mid')
            self.proxy.add_spike_train_plot(self.pop_learning_out, label='Learning Out')

    def calc_reward(self, lanelet_information, angle):

        isOnLanelet = lanelet_information.is_on_lane
        reward = 0
        distance = lanelet_information.distance
        if isOnLanelet:
            reward = (1-distance)
        else:
            if (lanelet_information.is_left and angle < 0) or (lanelet_information.is_right and angle > 0):
                reward = 0.1
            else:
                reward = -10. * (distance + 1) * (abs(angle)+1)

        if reward < -20:
            reward = -20

        return reward

    def inject(self, frame, lanelet_information):
        frame_l = frame[0:50, 0:50]
        frame_r = frame[0:50, 50:100]

        self.pop_in_l.set(i_offset=frame_l.astype(float).flatten())
        self.pop_in_r.set(i_offset=frame_r.astype(float).flatten())

        tstop = 100.0
        nest.run(tstop)
        nest.end()

        if self.plot:
            self.proxy.update(nest.get_current_time())

        self.spikes[:, 0] = nest.nest.GetStatus(self.all_detectors, 'n_events')

        nest.nest.SetStatus(self.all_detectors, 'n_events', 0)

        self.spikes = np.roll(self.spikes, -1, axis=1)

        # print 'spikes', self.spikes
        if lanelet_information is not None:
            distance = lanelet_information.distance
        else:
            distance = -1

        num_spikes_l = self.spikes[-1,-1]
        num_spikes_r = self.spikes[-2,-1]

        num_spikes_diff = num_spikes_l - num_spikes_r
        # TODO ensure -1 <= angle <= 1
        angle = num_spikes_diff / 10 # minus = rechts
        brake = 0  # np.exp(abs(angle)) - 1
        gas = 1 / (abs(angle) + 2.5)
        print 'l {:3d} | r {:3d} | diff {:3d} | gas {:2.2f} | brake {:2.2f} | steer {:2.2f} | distance {:3.4f} |reward {:3.2f}'.format(
            num_spikes_l,
            num_spikes_r,
            num_spikes_diff,
            gas,
            brake,
            angle,
            distance,
            self.reward)

        return gas, brake, angle

    def calc_eligibility_change(self):
        # Only check if neurons spiked or not
        spikes = self.spikes.copy()
        spikes[spikes >= 1] = 1

        # Arrange the NUM_NEURONS presynaptic spikes for each neuron at each time step (except last time step).
        presynaptic = np.tile(spikes[..., :-1], (NUM_RL_NEURONS, 1, 1))
        # Exponential decay factor for postsynaptic spikes
        decay = np.exp(-np.arange(NUM_TRACE_STEPS - 1) * TIME_STEP / TAU)[::-1]
        change_elig = presynaptic * decay
        # Sum over time. Calculate the weighted incoming presynaptic spikes at each neuron connection.
        change_elig = np.sum(change_elig, axis=2)

        # Eligiblity sign depends on whether the postsynaptic neuron spikes.
        postsynaptic = np.tile(spikes[..., 0], (NUM_RL_NEURONS, 1)).T
        change_elig[postsynaptic == 1] *= BETA_SIGMA
        change_elig[postsynaptic == 0] *= -BETA_SIGMA * SIGMA / (1 - SIGMA)

        return change_elig[0:NUM_MIDDLE_LEARNING_LAYER,NUM_MIDDLE_LEARNING_LAYER:NUM_RL_NEURONS]
        # change_elig[i, j] is now the eligibility trace change for the connection from i to j
        # TODO: check if actual connection exists

    def reset_weights(self):
        oldlearn = self.learn
        self.learn = False

        random_distribution = RandomDistribution('uniform', [0.1, 2])
        self.projection_learning_out.setWeights(random_distribution)
        # initialize weights for learning
        i = 0
        for neuron in self.pop_learning_out:
            # Skip detector
            if i > 1:
                break
            target_con = nest.nest.GetConnections(target=[neuron])
            weights = nest.nest.GetStatus(target_con, 'weight')
            j = 0
            for w in weights:
                self.weights[j,i] = w
                j += 1
            i += 1

        self.learn = oldlearn

    def update_weights(self, lanlet_information, angle):

        self.reward = self.calc_reward(lanlet_information, angle)

        self.eligibility_trace *= DISCOUNT_FACTOR
        self.eligibility_trace += self.calc_eligibility_change()

        # print 'weights:',self.weights
        # print 'elig:',self.eligibility_trace
        # print 'reward:',reward

        self.learningrate_fakt -= self.learningrate_fakt/100
        self.weights += self.learningrate_fakt * self.reward * self.eligibility_trace
        #self.weights += LEARNING_RATE * self.reward * self.eligibility_trace
        self.projection_learning_out.setWeights(self.weights / 1000)

    def reset_car(self):
        rospy.wait_for_service('reset_car')
        try:
            reset_car_call = rospy.ServiceProxy('reset_car', reset_car)
            pose = reset_car_call(0)
            print pose
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e


class SizedTimedRotatingFileHandler(handlers.TimedRotatingFileHandler):
    """
    Handler for logging to a set of files, which switches from one file
    to the next when the current file reaches a certain size, or at certain
    timed intervals
    """
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None,
                 delay=0, when='h', interval=1, utc=False):
        if maxBytes > 0:
            self.mode = 'a'
        handlers.TimedRotatingFileHandler.__init__(
            self, filename, when, interval, backupCount, encoding, delay, utc)
        self.maxBytes = maxBytes

    def shouldRollover(self, record):
        if self.stream is None:                 # delay was set...
            self.stream = self._open()
        if self.maxBytes > 0:                   # are we rolling over?
            msg = "%s\n" % self.format(record)
            self.stream.seek(0, 2)  #due to non-posix-compliant Windows feature
            if self.stream.tell() + len(msg) >= self.maxBytes:
                return 1
        t = int(time.time())
        if t >= self.rolloverAt:
            return 1
        return 0


def create_argument_parser():
    parser = argparse.ArgumentParser(description='Spiking Neural Network Node')
    parser.add_argument('-p', '--plot', action='store_true')
    parser.add_argument('-l', '--log', action='store_true')
    return parser


def create_weights_logger(compressed = False, maxbytes=104857600):  # maxbytes default: 100 MB
    # Create timestamp for folder name in
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    dumpdir = os.path.join(WEIGHTS_DUMPS_DIR, timestamp)
    os.mkdir(dumpdir)

    ext = 'bz2' if compressed else 'txt'
    filename = os.path.join(dumpdir, 'weights.' + ext)

    logger = logging.getLogger('WeightsLogger')
    logger.setLevel(logging.INFO)
    encoding = 'bz2' if compressed else None
    handler = SizedTimedRotatingFileHandler(filename, maxBytes=maxbytes, backupCount=5, when='D', interval=10, encoding=encoding)

    logger.addHandler(handler)

    return logger


def log_weights(logger, current_time, weights, formatstring='%.3f'):
    # time, w1, w2, ...
    s = (formatstring % current_time) + "," + ",".join(map(lambda x: formatstring % x, weights.flatten()))
    logger.info(s)


class CockpitViewModel:
    def __init__(self, net):
        self.view = Cockpit(self)

        # Traced variables
        self.learn = None
        self.weights_mean_left = None
        self.weights_mean_right = None

        # Important: Start view before initializing the variables
        self.view.start()
        self.net = net

    def initializeViewModel(self):
        self.learn = Tk.BooleanVar()
        self.learn.set(self.net.learn)
        self.weights_mean_left = Tk.StringVar()
        self.weights_mean_right = Tk.StringVar()

        self.learn.trace("w", self.learn_changed)

    def update_weights_mean(self):
        formatstring = "%.4f"
        weights_mean = np.mean(self.net.weights, axis=0)
        self.weights_mean_left.set(formatstring % weights_mean[0])
        self.weights_mean_right.set(formatstring % weights_mean[1])

    def reset_car_command(self):
        self.net.reset_car()

    def reset_weights_command(self):
        self.net.reset_weights()

    def learn_changed(self, *args):
        self.net.learn = not self.net.learn


class Cockpit(threading.Thread):
    def __init__(self, viewmodel):
        threading.Thread.__init__(self)
        self.viewmodel = viewmodel
        self.root = None
        self.leftWeightsMeanLabel = None
        self.rightWeightsMeanLabel = None

    def callback(self):
        self.root.quit()

    def update_weights_mean(self, weights_mean):
        self.leftWeightsMeanLabel.labelText = "%.4f" % weights_mean[0]
        self.rightWeightsMeanLabel.labelText = "%.4f" % weights_mean[1]


    def run(self):
        self.root = Tk.Tk()
        self.root.wm_title("Cockpit")

        self.viewmodel.initializeViewModel()
        self.root.protocol("WM_DELETE_WINDOW", self.callback)

        # Create GUI Elements
        learnCheckbutton = Tk.Checkbutton(self.root, text="Learn", var=self.viewmodel.learn)
        resetCar = Tk.Button(self.root, text = "Reset Car", command=self.viewmodel.reset_car_command)
        constantWeightsTextBox = Tk.Text(self.root, height=1, width=20)
        resetWeights = Tk.Button(self.root, text="Reset Weights", command=self.viewmodel.reset_weights_command)

        self.leftWeightsMeanLabel = Tk.Label(self.root, text="left", textvariable=self.viewmodel.weights_mean_left)
        self.rightWeightsMeanLabel = Tk.Label(self.root, text="right", textvariable=self.viewmodel.weights_mean_right)

        # Arrange GUI Elements
        learnCheckbutton.grid(row=0, column=0)
        resetCar.grid(row=1, column=0)
        resetWeights.grid(row=2, column=0)
        # constantWeightsTextBox.grid(row=2, column=1)
        self.leftWeightsMeanLabel.grid(row=3, column=0)
        self.rightWeightsMeanLabel.grid(row=3, column=1)

        self.root.mainloop()

def _main(argv):
    parser = create_argument_parser()
    n = parser.parse_args()

    node = SpikingNetworkNode()
    w, h = node.last_frame.shape

    network = BraitenbergNetwork(timestep=TIME_STEP, simduration=20, width=w, height=h)
    world = World()
    learner = ReinforcementLearner(network, world, BETA_SIGMA, SIGMA, TAU, NUM_TRACE_STEPS, 2, DISCOUNT_FACTOR,
                                   LEARNING_RATE)
    plotter = NetworkPlotter(network)

    while True:
        frame = node.last_frame
        network.inject_frame(frame, node.lanelet_information)
        network.simulate()
        actions = network.decode_actions()

        gas = actions['gas']
        brake = actions['brake']
        steering_angle = actions['steering_angle']

        learner.learn()

        plotter.update()

        frame = frame.copy()
        frame.T[50] = 255 - frame.T[50]
        cv2.imshow('Cam', frame)
        cv2.waitKey(1)

        node.publish(gas, brake, steering_angle)


def main(argv):
    parser = create_argument_parser()
    n = parser.parse_args()

    node = SpikingNetworkNode()
    w, h = node.last_frame.shape

    net = SpikingNetwork(w, h, n.plot)

    window = cv2.namedWindow('Cam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, w * 4, h * 4)

    cockpit = CockpitViewModel(net)


    if n.log:
        compressed = False
        maxbytes = 100 * 2**10
        logger = create_weights_logger(compressed,maxbytes)
        logperiod = 15 * 60 # log every 'logPeriod' seconds

        starttime = time.time()
        periodstarttime = starttime

    while True:
        if node.is_set_back:
            net.spikes = np.zeros((NUM_RL_NEURONS, NUM_TRACE_STEPS), dtype='int32')
            net.eligibility_trace = np.zeros((NUM_MIDDLE_LEARNING_LAYER,NUM_OUT_LEARNING_LAYER))

        if n.log:
            now = time.time()
            timepassed = now - starttime
            timepassedperiod = now - periodstarttime

            if timepassedperiod >= logperiod:
                log_weights(logger, timepassed, net.weights)
                periodstarttime = time.time()

        node.is_set_back = False
        cv2.imshow('weights',cv2.resize(net.weights / 1000 ,(0,0), fx=100, fy=10,interpolation=cv2.INTER_NEAREST))
        frame = node.last_frame
        gas, brake, angle = net.inject(frame, node.lanelet_information)
        frame2 = frame.copy()
        frame2.T[50] = 255 - frame2.T[50]
        cv2.imshow('Cam', frame2)
        cv2.waitKey(1)

        node.publish(gas, brake, angle)

        if net.learn:
            net.update_weights(node.lanelet_information, angle)
            #cockpit.update_weights_mean()


if __name__ == '__main__':
    print "hello"
    main(sys.argv)