#!/bin/bash
source /disk/users/mlprak2/.bashrc
set -x
echo $(pwd)

CATKIN_SRC="/disk/users/mlprak2/robot_folders/checkout/ml-praktikum/catkin_workspace/src/kog-auto-ws-2015/vehicle_control/src"

echo "Copying files ..."
cp lanelet_information.cpp $CATKIN_SRC/
cp lanelet_utils.* $CATKIN_SRC/
cp lanelet_random_pos.cpp $CATKIN_SRC/

echo "Copying Service Files ..."
cp ../srv/*.srv random_pos_service.srv $CATKIN_SRC/../srv
cp *.launch /disk/users/mlprak2/robot_folders/checkout/ml-praktikum/catkin_workspace/src/kog-auto-ws-2015/cvs_gazebo/launch

cd /disk/users/mlprak2/robot_folders/checkout/ml-praktikum/catkin_workspace
catkin_make install --use-ninja
