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
import cockpit


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
        self.angle_vehicle_lane = data[2]

        self.is_on_lane = bool(data[1])
        self.is_left = bool(data[3])
        self.is_right = not self.is_left

    def __str__(self):
        s = """
        Is on Right Lane: %s
        Is Left: %s
        Is Right: %s
        Distance: %.2f
        Angle of Vehicle to Lanelet (deg): %.2f deg
        """ % (self.is_on_lane, self.is_left, self.is_right, self.distance, self.angle_vehicle_lane/np.pi * 180)

        return s


class Learner:
    __metaclass__ = ABCMeta

    def __init__(self, network, world):
        self._network = network
        self._world = world

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, value):
        self._network = value

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

        self._spikes = {}
        self._neurons = set([])
        self._weights = None
        self._connections = None

        self._prepare_learner()

    def _prepare_learner(self):
        if self.network is not None:
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
                self._spikes[neuron] = np.zeros(self._steps_to_trace + 1)

    @Learner.network.setter
    def network(self, value):
        self._network = value
        self._prepare_learner()

    def learn(self):
        self._update_spikes()
        self._update_weights()

        print "z: ", self._eligibility_trace
        print "w: ", self._weights

    def _update_spikes(self):
        # TODO: This might be slower than previous implementation. It would be better to sort
        # TODO: ... the neurons and use the permutation for a matrix instead of looking up stuff in a dictionary
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

    def __init__(self, timestep, simduration, learner, should_learn=True, image_topic='/spiky/retina_image'):

        self._timestep = timestep
        self._simduration = simduration
        self._detectors = {}
        self._weights = None
        self._output_pop = None
        self._plastic_connections = None
        self._last_action = 1
        self._learner = learner
        self._postsynaptic_learning_neurons = []

        if self._learner is not None:
            self._learner.network = self

        self._should_learn = should_learn

        # Populations
        self.pop_input_image_r = None
        self.pop_input_image_l = None
        self.pop_encoded_image_l = None
        self.pop_encoded_image_r = None

        self.spikedetector_enc_image_l = None
        self.spikedetector_enc_image_r = None

        # Preparing sensory information subscriber
        rospy.init_node('SNN', disable_signals=True)
        self._bridge = CvBridge()
        self._last_frame = None
        self._retina_image_subscriber = rospy.Subscriber(image_topic, Image, self._handle_frame, queue_size=1)
        rospy.wait_for_message('/spiky/retina_image', Image)

        if self._last_frame is not None:
            shape = np.shape(self._last_frame)
            self._width = shape[0]
            self._height = shape[1]

        self._car_update_publisher = rospy.Publisher('/AADC_AudiTT/carUpdate', Vector3, queue_size=1)

        self._build_input_layer()
        self._build_output_layer()
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

        # bild seiten staerker gewichtet
        self.projection_out_l[0]._set_weight(2.0)
        self.projection_out_r[1]._set_weight(2.0)
        self.projection_out_l[2]._set_weight(4.0)
        self.projection_out_r[3]._set_weight(4.0)

    def _build_output_layer(self):
        self._output_pop = Population(2, IF_curr_alpha, {'tau_refrac': 0.1, 'i_offset': np.zeros(2)})

    def _handle_frame(self, frame):
        self._last_frame = self._bridge.imgmsg_to_cv2(frame)

    def populate_plotter(self, plotter):
        plotter.add_spike_train_plot(self.pop_encoded_image_l, label='Input Left')
        plotter.add_spike_train_plot(self.pop_encoded_image_r, label='Input Right')

    def _create_spike_detectors(self):
        self.spikedetector_enc_image_l = nest.nest.Create('spike_detector', params={'withgid': True, 'withtime': True})[0]
        self.spikedetector_enc_image_r = nest.nest.Create('spike_detector', params={'withgid': True, 'withtime': True})[0]

        nest.nest.Connect(self.pop_encoded_image_l[0], self.spikedetector_enc_image_l)
        nest.nest.Connect(self.pop_encoded_image_r[0], self.spikedetector_enc_image_r)

        self.detectors[self.pop_encoded_image_l[0]] = self.spikedetector_enc_image_l
        self.detectors[self.pop_encoded_image_r[0]] = self.spikedetector_enc_image_r

        for neuron in self._output_pop:
            detector = nest.nest.Create('spike_detector', params={'withgid': True, 'withtime': True})[0]
            self.detectors[neuron] = detector
            nest.nest.Connect(neuron, detector)

    @property
    def should_learn(self):
        return self._should_learn

    @should_learn.setter
    def should_learn(self, value):
        self._should_learn = value

    @property
    def last_frame(self):
        return self._last_frame

    @property
    def learner(self):
        return self._learner

    @learner.setter
    def learner(self, value):
        self._learner = value
        if self._learner is not None:
            self._learner.network = self

    @property
    def encoded_image_pops(self):
        return {'left': self.pop_encoded_image_l, 'right': self.pop_encoded_image_r}

    @property
    def plotter(self):
        return self._plotter

    @property
    def output_pop(self):
        return self._output_pop

    @property
    def timestep(self):
        return self._timestep

    @property
    def simduration(self):
        return self._simduration

    @property
    def detectors(self):
        return self._detectors

    @property
    def plastic_connections(self):
        if self._plastic_connections is None:
            self._plastic_connections = nest.nest.GetConnections(target=self.postsynaptic_learning_neurons)
        return self._plastic_connections


    def get_weights(self):
        connections = self.plastic_connections
        return np.array(nest.nest.GetStatus(connections, 'weight'))

    def set_weights(self, weights):
        connections = self.plastic_connections
        nest.nest.SetStatus(connections, [{'weight': w} for w in weights])

    def reset_weights(self):
        weights = np.random.uniform(-0.1, 1.0, len(self.plastic_connections)) * 5000
        self.set_weights(weights)

    def decode_actions(self):
        spikes = np.array(nest.nest.GetStatus([self.detectors[x] for x in self.output_pop], 'n_events'))

        # last two are output neurons
        num_spikes_l = spikes[0]
        num_spikes_r = spikes[1]

        num_spikes_diff = np.power(num_spikes_l, 2) - np.power(num_spikes_r, 2)
        angle = num_spikes_diff/ 10  # minus = rechts
        brake = 0
        #gas = 1.0 / (np.sqrt(abs(angle)) + 2.5)
        gas = np.max(((1 - np.abs(angle))*0.5, 0.2))
        actions = {'gas': gas, 'brake': brake, 'steering_angle': angle}

        return actions

    @property
    def postsynaptic_learning_neurons(self):
        return self._postsynaptic_learning_neurons

    def step(self):
        self.inject_frame(self._last_frame)
        self.simulate()

        if self.should_learn:
            if self._learner is not None:
                self._learner.learn()
            else:
                print "WARNING: should_learn = true but learner is None"

        self.act()

    def act(self):
        actions = self.decode_actions()
        gas = actions['gas']
        brake = actions['brake']
        steering_angle = actions['steering_angle']
        self._car_update_publisher.publish(gas, brake, steering_angle)
        self._last_action = gas ##

    def inject_frame(self, frame):
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
    def __init__(self, timestep, simduration, learner, should_learn=True, image_topic='/spiky/retina_image'):
        super(BraitenbergNetwork, self).__init__(timestep, simduration, learner, should_learn, image_topic)

        self._left_connection = None
        self._right_connection = None

        self._build_network()

    def _build_network(self):

        l = self.encoded_image_pops['left']
        r = self.encoded_image_pops['right']

        self._left_connection = Projection(l, self.output_pop, AllToAllConnector())
        self._right_connection = Projection(r, self.output_pop, AllToAllConnector())

        # Here go the neurons whose presynaptic connections are
        # adapted by a learner. In this case, we only want to adapt
        # the presynaptic connections of the output neurons.
        for neuron in self.output_pop:
            self.postsynaptic_learning_neurons.append(neuron)

        self.reset_weights()

    def populate_plotter(self, plotter):
        super(BraitenbergNetwork, self).populate_plotter(plotter)
        plotter.add_spike_train_plot(self.output_pop, 'Output R/L')

    def reset_weights(self):
        weights = np.array([1.0, -0.3, -0.3, 1.0])*3000
        self.set_weights(weights)


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

    @abstractproperty
    def last_reward(self):
        pass


class World(BaseWorld):
    def __init__(self,topic='/laneletInformation'):
        super(World, self).__init__()
        self.sub_lanelet = rospy.Subscriber(topic, Float64MultiArray, self._update_state,
                                            queue_size=1)
        self._state = None
        self._last_reward = None
        self._last_distance = None
        self._last_angle_vehicle_lane = None

    def calculate_reward(self, actions):
        state = self._state

        if state is None:
            return 0

        steering_angle = actions['steering_angle']
        is_on_lanelet = state.is_on_lane
        angle_vehicle_lane = state.angle_vehicle_lane

        # distance from the center of the right lane
        distance = state.distance * 2
        distance_reward = 1 - distance

        # angle_vehicle_lane_reward = np.pi/2 - np.abs(angle_vehicle_lane)

        if self._last_angle_vehicle_lane is not None:
            angle_vehicle_lane_reward = np.abs(self._last_angle_vehicle_lane) - np.abs(angle_vehicle_lane)
        else:
            angle_vehicle_lane_reward = 0

        # if is_on_lanelet:
        #     reward = (1 - distance)
        # else:
        #     if (state.is_left and steering_angle < 0) or (state.is_right and steering_angle > 0):
        #         reward = 0.1
        #     else:
        #         reward = -10. * (distance + 1) * (abs(steering_angle) + 1)

        reward = distance_reward + 0.5 * angle_vehicle_lane_reward

        if reward < -1:
            reward = -1

        self._last_reward = reward * 3
        self._last_angle_vehicle_lane = angle_vehicle_lane
        self._last_distance = distance

        return self._last_reward

    @property
    def current_state(self):
        return self._state

    @property
    def last_reward(self):
        return self._last_reward

    def _update_state(self, state):
        if len(state.data) > 0:
            self._state = LaneletInformation(state.data)


class DeepNetwork(BaseNetwork):
    def __init__(self, timestep, simduration, learner, number_middle_layers, number_neurons_per_layer,
                 should_learn=True, image_topic='/spiky/retina_image'):
        super(DeepNetwork, self).__init__(timestep, simduration, learner, should_learn, image_topic)

        self._middle_pops = []

        self._number_neurons_per_layer = number_neurons_per_layer
        self._num_out_learning = len(self.output_pop)
        self._number_middle_layers = number_middle_layers

        self._build_network()

    def _build_network(self):
        l = self.encoded_image_pops['left']
        r = self.encoded_image_pops['right']

        # Appends variable number of middle layers
        for i in range(self._number_middle_layers):
            current_pop = Population(self._number_neurons_per_layer, IF_curr_alpha,
                                             {'tau_refrac': 0.1, 'i_offset': np.zeros(self._number_neurons_per_layer)})
            self._middle_pops.append(current_pop)
            if i == 0:
                # Connect input layer to first middle layer
                left_in_connection = Projection(l, self._middle_pops[i], AllToAllConnector())
                right_in_connection = Projection(r, self._middle_pops[i], AllToAllConnector())
            else:
                # Connect latest middle layer to current
                connection = Projection(self._middle_pops[i - 1], self._middle_pops[i], AllToAllConnector())

            for neuron in current_pop:
                detector = nest.nest.Create('spike_detector', params={'withgid': True, 'withtime': True})[0]
                self.detectors[neuron] = detector
                nest.nest.Connect(neuron, detector)

        # Connect last middle layer to output
        left_out_connection = Projection(self._middle_pops[self._number_middle_layers - 1], self.output_pop,
                                         AllToAllConnector())
        right_out_connection = Projection(self._middle_pops[self._number_middle_layers - 1], self.output_pop,
                                          AllToAllConnector())

        # Here go the neurons whose presynaptic connections are
        # adapted by a learner. In this case, we only want to adapt
        # the presynaptic connections of the output neurons.
        for pop in self._middle_pops:
            for neuron in pop:
                self.postsynaptic_learning_neurons.append(neuron)

        for neuron in self.output_pop:
            self.postsynaptic_learning_neurons.append(neuron)

        self.reset_weights()

    def reset_weights(self):
        weights = np.random.uniform(-1.5,2,len(self.plastic_connections))*2500
        weights_inout = np.zeros(((NUM_OUT_LEARNING_LAYER+NUM_IN_LEARNING_LAYER)*self._number_neurons_per_layer), dtype=Float64MultiArray)
        counter = 0
        for i in range(NUM_OUT_LEARNING_LAYER+NUM_IN_LEARNING_LAYER):
            temp_weights = (np.random.dirichlet(np.ones(self._number_neurons_per_layer),size=1))*100
            temp_weights_avg = np.average(temp_weights)
            temp_weights[:] = [x - temp_weights_avg for x in temp_weights]
            temp_weights *= 250
            for j in range(len(temp_weights[0])):
                weights_inout[counter] = temp_weights[0][j]
                counter +=1


        self.set_weights(weights, weights_inout)

    def set_weights(self, weights, weights_inout):
        num_connections_in = NUM_IN_LEARNING_LAYER * self._number_neurons_per_layer
        num_connections_out = NUM_OUT_LEARNING_LAYER * self._number_neurons_per_layer
        connections = self.plastic_connections
        connections_inout = []
        for con in self.plastic_connections[:num_connections_in]:
            connections_inout.append(con)
        for con in self.plastic_connections[-num_connections_out:]:
            connections_inout.append(con)

        nest.nest.SetStatus(connections, [{'weight': w} for w in weights])
        nest.nest.SetStatus(connections_inout, [{'weight': w} for w in weights_inout])

    def populate_plotter(self, plotter):
        super(DeepNetwork, self).populate_plotter(plotter)
        plotter.add_spike_train_plot(self.output_pop, 'Output R/L')
        i = 0
        for pop in self._middle_pops:
            plotter.add_spike_train_plot(pop, 'Middle Pop %d' % i)
            i += 1


class NetworkPlotter:
    def __init__(self, network, plot_steps=10):
        self._network = network
        self._plotter = PlotterProxy(network.simduration, plot_steps)
        network.populate_plotter(self._plotter)

    def update(self):
        self._plotter.update(nest.get_current_time())


class SizedTimedRotatingFileHandler(handlers.TimedRotatingFileHandler):
    """
    Handler for logging to a set of files, which switches from one file
    to the next when the current file reaches a certain size, or at certain
    timed intervals
    """
    def __init__(self, filename, mode='a', max_bytes=0, backup_count=0, encoding=None,
                 delay=0, when='h', interval=1, utc=False):
        if max_bytes > 0:
            self.mode = 'a'
        handlers.TimedRotatingFileHandler.__init__(
            self, filename, when, interval, backup_count, encoding, delay, utc)
        self.maxBytes = max_bytes

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


class NetworkLogger:
    def __init__(self, network, formatstring='%.3f', log_period=900, maxbytes=104857600, compressed=False, when='D',
                 interval=10, backup_count=5):
        self._network = network

        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        dumpdir = os.path.join(WEIGHTS_DUMPS_DIR, timestamp)
        if not os.path.isdir(dumpdir):
            if not os.path.isdir(DUMPS_DIR):
                os.mkdir(DUMPS_DIR)
                os.mkdir(WEIGHTS_DUMPS_DIR)

            os.mkdir(dumpdir)

        ext = 'bz2' if compressed else 'txt'
        weights_filename = os.path.join(dumpdir, 'weights.' + ext)
        reward_filename = os.path.join(dumpdir, 'reward.' + ext)

        self._reward_logger = logging.getLogger("RewardLogger")
        self._reward_logger.setLevel(logging.INFO)

        self._weights_logger = logging.getLogger('WeightsLogger')
        self._weights_logger.setLevel(logging.INFO)

        encoding = 'bz2' if compressed else None
        weights_handler = SizedTimedRotatingFileHandler(weights_filename, max_bytes=maxbytes, backup_count=backup_count, when=when,
                                                interval=interval, encoding=encoding)

        self._weights_logger.addHandler(weights_handler)

        reward_handler = SizedTimedRotatingFileHandler(reward_filename, max_bytes=maxbytes, backup_count=backup_count, when=when,
                                                interval=interval, encoding=encoding)
        self._reward_logger.addHandler(reward_handler)
        self._reward = []
        self._starttime = time.time()
        self._period_starttime = self._starttime
        self._formatstring = formatstring
        self._log_period = log_period

    def log(self):
        now = time.time()
        actions = self._network.decode_actions()
        self._reward.append(self._network.learner.world.calculate_reward(actions))

        if now - self._period_starttime > self._log_period:
            self.log_weights(now)
            self.log_reward(now)
            self._period_starttime = now

    def log_reward(self, now):
        mean_reward = np.mean(np.array(self._reward))
        self._reward = []

        self._reward_logger.info('%f, %f' % (now, mean_reward))

    def log_weights(self, now):
            current_time = now - self._starttime
            weights = self._network.get_weights()

            s = (self._formatstring % current_time) + "," \
                + ",".join(map(lambda x: self._formatstring % x, weights.flatten()))
            self._weights_logger.info(s)


class Configuration:
    def __init__(self):
        self.beta = 0


def create_argument_parser():
    parser = argparse.ArgumentParser(description='Spiking Neural Network Node')
    parser.add_argument('-p', '--plot', action='store_true')
    parser.add_argument('-l', '--log', action='store_true')
    return parser


def main(argv):
    parser = create_argument_parser()
    n = parser.parse_args()

    world = World()

    #network = BraitenbergNetwork(timestep=TIME_STEP, simduration=20, learner=None, should_learn=False)
    network = DeepNetwork(timestep=TIME_STEP, simduration=10, learner=None, should_learn=False,
                              number_middle_layers=2, number_neurons_per_layer=10)

    learner = ReinforcementLearner(network, world, BETA_SIGMA, SIGMA, TAU, NUM_TRACE_STEPS, 2, DISCOUNT_FACTOR,
                                   LEARNING_RATE)
    network.learner = learner
    n.plot = True
    if n.plot:
        plotter = NetworkPlotter(network, plot_steps=20)

    n.log = True
    if n.log:
        logger = NetworkLogger(network, log_period=10)

    cockpit_view = cockpit.CockpitViewModel(network)

    while True:
        # Inject frame to network and start simulation
        network.step()

        if n.plot:
            plotter.update()

        if n.log:
            logger.log()

        cockpit_view.update()

        frame = network.last_frame.copy()
        frame.T[50] = 255 - frame.T[50]
        cv2.imshow('Cam', frame)
        cv2.waitKey(1)

if __name__ == '__main__':
    print "hello"
    main(sys.argv)