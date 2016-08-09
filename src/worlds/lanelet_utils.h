#ifndef LANELET_UTILS_H
#define LANELET_UTILS_H

//ROS
#include "ros/ros.h"
#include "geometry_msgs/Pose.h"

//Lanelet
#include "liblanelet/LaneletMap.hpp"
#include "liblanelet/lanelet_point.hpp"
#include "liblanelet/Lanelet.hpp"

#define GAZEBO_Z 0.449960

LLet::point_with_id_t transformPointToGPS(geometry_msgs::Point point, LLet::point_with_id_t reference_point);

geometry_msgs::Point transformGPSToPoint(LLet::point_with_id_t point, LLet::point_with_id_t reference_point);

geometry_msgs::Point quatRotation(const geometry_msgs::Point& p, const geometry_msgs::Quaternion& q);

std::string retrieveLaneletFilename(void);

#endif // LANELET_UTILS_H
