#include "ros/ros.h"
#include "liblanelet/LaneletMap.hpp"
#include "liblanelet/lanelet_point.hpp"
#include "liblanelet/Lanelet.hpp"
#include <iostream>
#include "geometry_msgs/Pose.h"
#include <sstream>
#include "std_msgs/String.h"

namespace _deprecated {

geometry_msgs::Point lastPos;
geometry_msgs::Quaternion lastQuat;
bool poseUpdated = false;
bool init = false;

void chatterCallback(const geometry_msgs::Pose::ConstPtr& msg)
{
  init = true;
  poseUpdated = true;
  lastPos = msg->position;
  lastQuat = msg->orientation;
}

LLet::point_with_id_t transformPointToGPS(geometry_msgs::Point point, LLet::point_with_id_t reference_point)
{
  /*
   * The coordinate system of the lanelet-map generated by the LaneletPlugin
   * is rotated and flipped in comparison to the coordinate system used by
   * Gazebo. Therefore we need to adjust the point before converting to
   * latitude and longitude.
   */
  LLet::point_xy_t xy_position(boost::make_tuple(point.y, -point.x));

  // Use the adjusted point and the reference point to get the GPS
  // coordinates.
  return LLet::from_vec(reference_point, xy_position);
}

geometry_msgs::Point transformGPSToPoint(LLet::point_with_id_t point, LLet::point_with_id_t reference_point)
{
  /*
   * The coordinate system of the lanelet-map generated by the LaneletPlugin
   * is rotated and flipped in comparison to the coordinate system used by
   * Gazebo. Therefore we need to adjust the point before converting from
   * latitude and longitude.
   */
  LLet::point_xy_t xy_position = LLet::vec(point, reference_point);
  geometry_msgs::Point p;
  p.x = boost::get<1>(xy_position);
  p.y = -boost::get<0>(xy_position);
  p.z = 0.449960; //adjust according to the Gazebo world and model

  return p;
}

int main(int argc, char **argv)
{
  if (3 != argc)
  {
    std::cout << "Not enough or invalid arguments, please try again.\n";
    sleep(2000);
    exit(0);
  }

  double MAX_DISTANCE = 100.0;
  LLet::point_with_id_t reference_point(boost::make_tuple(0.0, 0.0, -1));

  /// TODO: Possibly better to create a node that indicates the .osm-File to open
  std::string fileName = "/disk/users/mlprak2/lea_fabian/ml2-spiking/src/worlds/onecrossing_lanelet.osm"; //use an appropriate .osm file
  LLet::LaneletMap llmap(fileName);

  ros::init(argc, argv, "drive");
  ros::NodeHandle n;
  ros::Publisher chatter_pub = n.advertise<std_msgs::String>("/AADC_AudiTT/DistanceOneCrossing", 1);

  ros::Rate loop_rate(2); // 2 updates per second
  ros::Subscriber sub = n.subscribe("/AADC_AudiTT/carPose", 1000, chatterCallback);

  LLet::point_with_id_t current_location;
  std::ostringstream strs;

  while (ros::ok())
  {
    if (!init)
    {
      ros::spinOnce();
      loop_rate.sleep();
      continue;
    }

    if (poseUpdated)
    {
      // Take the pose from Gazebo and transform it to GPS coordinates for the LaneletMap.
      current_location = transformPointToGPS(lastPos, reference_point);

      // Try to find lanelet closest to our current location within a maximum distance.
      LLet::lanelet_ptr_t llnet;
      llmap.map_matching(current_location, llnet, MAX_DISTANCE);

      if (llnet != NULL)
      {
        // With a lanelet and a current position, we can call various functions
        // (see Lanelet.hpp or LaneletMap.hpp)
        // For example:
        double distance = llnet->distance_from_center_line_to(current_location);
        ROS_INFO_STREAM("DISTANCE TO CENTER LINE: " << distance);
        distance = llnet->distance_to(current_location);
        ROS_INFO_STREAM("DISTANCE TO LANELET: " << distance);
        bool isCovered = llnet->covers_point(current_location);
        ROS_INFO_STREAM("LANELET COVERS CURRENT LOCATION " << isCovered);        
        float angle = 0.0;
        ROS_INFO_STREAM("ANGLE TO LANELET: " << angle);
        strs.str("");
        strs << distance;
      }
      else
      {
        // Handle this case!
        ROS_INFO("No suitable lanelet found!");
      }
      poseUpdated = false;
    }

    std_msgs::String msg;
    msg.data = strs.str();
    chatter_pub.publish(msg);

    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
}
