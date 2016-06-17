#!/usr/bin/env bash

source kill_gazebo.sh
source ~/robot_folders/checkout/ml-praktikum/catkin_workspace/install/share/cvs_model_resources/setup.sh
roslaunch cvs_gazebo kog_auto_1516_nocrossings.launch
