#ifndef LANELET_UTILS_H
#define LANELET_UTILS_H

//ROS
#include "ros/ros.h"
#include "geometry_msgs/Pose.h"

//Lanelet
#include "liblanelet/LaneletMap.hpp"
#include "liblanelet/lanelet_point.hpp"
#include "liblanelet/Lanelet.hpp"

typedef LLet::point_with_id_t lanelet_point;

typedef struct side_information {
    bool isLeftFromMidLane;
    bool isOnLeftLane;
    bool isOnLane;
} side_information_t;

#define GAZEBO_Z 0.449960

//Coordinate transformation and other utility functions
LLet::point_with_id_t transformPointToGPS(geometry_msgs::Point point, LLet::point_with_id_t reference_point);
geometry_msgs::Point transformGPSToPoint(LLet::point_with_id_t point, LLet::point_with_id_t reference_point);
geometry_msgs::Point quaternionRotation(const geometry_msgs::Point& p, const geometry_msgs::Quaternion& q);
double determineSide(const LLet::point_with_id_t& v, LLet::lanelet_ptr_t llnet, double* angleVehicleLane = NULL);
boost::tuple<double, double> normalize(boost::tuple<double, double> v);
double length(boost::tuple<double, double> v);

//Conversion
geometry_msgs::Point geom_point(double x, double y, bool normalize = false);
geometry_msgs::Point geom_point(lanelet_point p, bool normalize = false);

//Retrieval of Lanelet Filename using ros params
std::string retrieveLaneletFilename(void);

#endif // LANELET_UTILS_H
