#!/bin/sh
CATKIN_SRC="/disk/users/mlprak2/robot_folders/checkout/ml-praktikum/catkin_workspace/src/kog-auto-ws-2015/vehicle_control/src"
cp lanelet_information.cpp $CATKIN_SRC/
cdros
catkin_make install --use-ninja
