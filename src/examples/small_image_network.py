# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:41:35 2016

@author: mlprak2
"""

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

setup(timestep=0.1)

image = cv2.imread('/fzi/ids/mlprak2/Bilder/impuls.tif');

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
frame = image

rates_init = np.zeros(frame.size)

pop_in = Population(frame.shape, SpikeSourcePoisson, {'rate': rates_init})
pop_out = Population(1, IF_curr_alpha, {'tau_refrac': 5 })

projection = Projection(pop_in, pop_out, AllToAllConnector())
projection.setWeights(1.0)

pop_in.set(rate=frame.astype(float).flatten())

pop_in.record('spikes')
pop_out.record('spikes')


tstop = 100.0
run(tstop)

spikes_in = pop_in.get_data()
data_out = pop_out.get_data()

for seg in data_out.segments:
    print seg    
    for st in seg.spiketrains:
        print st
        


for seg in spikes_in.segments:
    print seg    
    for st in seg.spiketrains:
        print st
