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
    std::string fileName = retrieveLaneletFilename();
    LLet::LaneletMap llmap(fileName);
    LLet::point_with_id_t reference_point(boost::make_tuple(0.0, 0.0, -1));
    size_t size = llmap.lanelets().size();
    int32_t random_l = rand() % size;

    ROS_INFO_STREAM("Size: " << size << ", Random Index: " << random_l);

    LLet::lanelet_ptr_t positionVec_ptr = llmap.lanelets()[random_l];
    LLet::strip_ptr_t center_strip = boost::get<LLet::SIDE::CENTER>(positionVec_ptr->bounds());
    std::vector<LLet::point_with_id_t> points = center_strip->pts();

    geometry_msgs::Point p1 = transformGPSToPoint(points[0], reference_point);
    geometry_msgs::Point p2 = transformGPSToPoint(points[1], reference_point);


    res.output.push_back(p1.x);
    res.output.push_back(p1.y);
    res.output.push_back(p1.z);
    res.output.push_back(p2.x);
    res.output.push_back(p2.y);
    res.output.push_back(p2.z);

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
