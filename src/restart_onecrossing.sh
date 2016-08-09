#!/usr/bin/env bash

source kill_gazebo.sh
source ~/robot_folders/checkout/ml-praktikum/catkin_workspace/install/share/cvs_model_resources/setup.sh
roslaunch cvs_gazebo onecrossing.launch
rosrun snn_plotter main
