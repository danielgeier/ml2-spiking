//ROS
#include "ros/ros.h"
#include "geometry_msgs/Pose.h"

//Lanelet
#include "liblanelet/LaneletMap.hpp"
#include "liblanelet/lanelet_point.hpp"
#include "liblanelet/Lanelet.hpp"

#include "lanelet_utils.h"

//Service
#include "vehicle_control/random_pos_service.h"

//C++-Libs
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>

bool add(vehicle_control::random_pos_service::Request &req, vehicle_control::random_pos_service::Response &res)
{
    LLet::point_with_id_t reference_point(boost::make_tuple(0.0, 0.0, -1));

    //Load the lanelet file
    std::string fileName = retrieveLaneletFilename();
    LLet::LaneletMap llmap(fileName);

    //Generate a random index
    size_t size = llmap.lanelets().size();
    int32_t random_l = rand() % size;
    ROS_INFO_STREAM("::Random Position::");
    ROS_INFO_STREAM("Size: " << size << ", Random Index: " << random_l);

    //Select lanelet with index generated in previous step and take 2 first points of the center line strip
    LLet::lanelet_ptr_t positionVec_ptr = llmap.lanelets()[random_l];
    LLet::strip_ptr_t center_strip = boost::get<LLet::SIDE::CENTER>(positionVec_ptr->bounds());
    std::vector<LLet::point_with_id_t> points = center_strip->pts();

    size = points.size();
    random_l = (rand() % size) - 1;
    random_l = random_l < 0? 0 : random_l;

    //Transform to Gazebo World Coordinates
    geometry_msgs::Point p1 = transformGPSToPoint(points[random_l], reference_point);
    geometry_msgs::Point p2 = transformGPSToPoint(points[random_l + 1], reference_point);
    geometry_msgs::Point heading;

    //Heading of the car in vector form
    heading.x = p2.x - p1.x;
    heading.y = p2.y - p1.y;
    heading.z = p2.z - p1.z;

    //Calculate the rotation angle with respect to the z-axis
    double angle = atan2(heading.y, heading.x);

    //Calculate the reset point (center of the first two line strip points)
    geometry_msgs::Point resetPoint;
    resetPoint.x = p1.x/2 + p2.x/2;
    resetPoint.y = p1.y/2 + p2.y/2;

    //Represent the heading of the car as a rotation around the z-axis by angle |angle|
    geometry_msgs::Quaternion rotationQuaternion;
    rotationQuaternion.x = 0;
    rotationQuaternion.y = 0;
    rotationQuaternion.z = sin(angle/2);
    rotationQuaternion.w = cos(angle/2);

    //Create the output
    res.output.push_back(resetPoint.x);
    res.output.push_back(resetPoint.y);

    res.output.push_back(rotationQuaternion.x);
    res.output.push_back(rotationQuaternion.y);
    res.output.push_back(rotationQuaternion.z);
    res.output.push_back(rotationQuaternion.w);

    return true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "provide random positions");
  ros::NodeHandle n;
  ros::ServiceServer service = n.advertiseService("random_pos_service", add);

  ROS_INFO("provide random positions.");
  ros::spin();

  return 0;
}
