from __future__ import division

import sys
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
import Tkinter as tk

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
        self.laneletAngle = data[2]

        self.isOnLane = bool(data[1])
        self.isLeft = bool(data[3])
        self.isRight = not self.isLeft

    def __str__(self):
        s = """
        Is on Lane: %s
        Is Left: %s
        Is Right: %s
        Distance: %.2f
        Angle of Vehicle to Lanelet (rad): %.2f
        """ % (self.isOnLane, self.isLeft, self.isRight, self.distance, self.laneletAngle)

        return s


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

        isOnLanelet = lanelet_information.isOnLane
        reward = 0
        distance = lanelet_information.distance
        if isOnLanelet:
            reward = (1-distance)
        else:
            if (lanelet_information.isLeft and angle < 0) or (lanelet_information.isRight and angle > 0):
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

        tstop = 100.0 #
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
        self.learn = tk.BooleanVar()
        self.learn.set(self.net.learn)
        self.weights_mean_left = tk.StringVar()
        self.weights_mean_right = tk.StringVar()

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
        self.root = tk.Tk()
        self.root.wm_title("Cockpit")

        self.viewmodel.initializeViewModel()
        self.root.protocol("WM_DELETE_WINDOW", self.callback)

        # Create GUI Elements
        learnCheckbutton = tk.Checkbutton(self.root, text="Learn", var=self.viewmodel.learn)
        resetCar = tk.Button(self.root, text = "Reset Car", command=self.viewmodel.reset_car_command)
        constantWeightsTextBox = tk.Text(self.root, height=1, width=20)
        resetWeights = tk.Button(self.root, text="Reset Weights", command=self.viewmodel.reset_weights_command)

        self.leftWeightsMeanLabel = tk.Label(self.root, text="left", textvariable=self.viewmodel.weights_mean_left)
        self.rightWeightsMeanLabel = tk.Label(self.root, text="right", textvariable=self.viewmodel.weights_mean_right)

        # Arrange GUI Elements
        learnCheckbutton.grid(row=0, column=0)
        resetCar.grid(row=1, column=0)
        resetWeights.grid(row=2, column=0)
        # constantWeightsTextBox.grid(row=2, column=1)
        self.leftWeightsMeanLabel.grid(row=3, column=0)
        self.rightWeightsMeanLabel.grid(row=3, column=1)

        self.root.mainloop()


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
            cockpit.update_weights_mean()


if __name__ == '__main__':
    main(sys.argv)