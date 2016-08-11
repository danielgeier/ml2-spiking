//ROS
#include "lanelet_utils.h"
#include "ros/ros.h"
#include "geometry_msgs/Pose.h"
#include "std_msgs/Float64MultiArray.h"

//Lanelet
#include "liblanelet/LaneletMap.hpp"
#include "liblanelet/lanelet_point.hpp"
#include "liblanelet/Lanelet.hpp"

//C++-Libs
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>

#define _USE_MATH_DEFINES 1 //needed on windows for usage of math constants such as M_PI
#define EPS 0.0001
#define MAX_DISTANCE 100.0


//Static variables to store car pose information etc.
geometry_msgs::Point positionVec;
geometry_msgs::Quaternion orientationQuat;
bool poseUpdated = false;
bool init = false;

void carPoseSubscriber_callback(const geometry_msgs::Pose::ConstPtr& msg)
{
  init = true;
  poseUpdated = true;
  positionVec = msg->position;
  orientationQuat = msg->orientation;
}


int main(int argc, char **argv)
{
  if (argc != 3)
  {
    std::cout << "Not enough or invalid arguments, please try again.\n";
    sleep(2000);
    exit(0);
  }

  LLet::point_with_id_t referencePoint(boost::make_tuple(0.0, 0.0, -1));

  //This direction - (1,0,0) - is transformed using the quaternion returned in car pose
  geometry_msgs::Point referenceDirection;
  referenceDirection.x = 1;
  referenceDirection.y = referenceDirection.z = 0;

  //Init ros
  ros::init(argc, argv, "drive");
  ros::NodeHandle nodeHandle;

  //Create Publisher
  const int ARRAY_SIZE = 10;
  ros::Publisher laneletPublisher = nodeHandle.advertise<std_msgs::Float64MultiArray>("laneletInformation", ARRAY_SIZE);  

  // 2 updates per second
  ros::Rate loop_rate(2);

  //Register subscribed nodes to callbacks
  ros::Subscriber carPoseSubscriber = nodeHandle.subscribe("/AADC_AudiTT/carPose", 1000, carPoseSubscriber_callback);

  //Current Vehicle Location on the lanelet world
  LLet::point_with_id_t current_location;

  //Load Lanelet
  std::string fileName = retrieveLaneletFilename();
  LLet::LaneletMap llmap(fileName);

  while (ros::ok())
  {
    ROS_INFO_STREAM("Using Lanelet file " << fileName);
    if (!init)
    {
      ros::spinOnce();
      loop_rate.sleep();
      continue;
    }

    //Prepare message to be sent via ROS-Node
    std_msgs::Float64MultiArray laneletInformationMessage;
    laneletInformationMessage.data.clear();

    if (poseUpdated)
    {
      // Take the pose from Gazebo and transform it to GPS coordinates for the LaneletMap.
      current_location = transformPointToGPS(positionVec, referencePoint);

      //Find out where the car is headed with respect to vector (1,0,0)
      geometry_msgs::Point vehicleHeadingGazebo = quaternionRotation(referenceDirection, orientationQuat);

      //The lanelet world is rotated by 90° -> rotate heading
      geometry_msgs::Point vehicleHeading;
      vehicleHeading.x = vehicleHeadingGazebo.y;
      vehicleHeading.y = -vehicleHeadingGazebo.x;

      //atan2 adapts the angle's sign depending on the quadrant to which the vector is pointing
      double vehicleHeadingAngle = std::atan2(vehicleHeading.y, vehicleHeading.x);

      //Lanelet heading output
      double laneletHeading = 0;
      LLet::lanelet_ptr_t llnet;
      llmap.map_matching(current_location, llnet, MAX_DISTANCE, &laneletHeading, true, vehicleHeadingAngle, M_PI/8);

      if (llnet != NULL)
      {
        //Computation
        double distance_center_line = llnet->distance_from_center_line_to(current_location);
        double distance = llnet->distance_to(current_location);

        LLet::strip_ptr_t centerLineStrip = boost::get<LLet::SIDE::CENTER>(llnet->bounds());
        double signedDistance = centerLineStrip->signed_distance(current_location);
        double side = determineSide(current_location,llnet);
        double mySignedDistance = side * distance_center_line;

        //Check the nearest lanelet to determine whether the car is on the lane or not
        bool isCovered = llnet->covers_point(current_location);

        //Determine on which side of the (constructed) mid lane the vehicle is driving
        bool isLeft  = side < 0;
        bool isRight = !isLeft;

        //Fill Message Array
        laneletInformationMessage.data.push_back(distance);
        laneletInformationMessage.data.push_back(isCovered);
        laneletInformationMessage.data.push_back(laneletHeading);
        laneletInformationMessage.data.push_back(isLeft);

        //Output
        ROS_INFO_STREAM("DISTANCE TO CENTER LINE: " << distance_center_line);
        ROS_INFO_STREAM("DISTANCE TO LANELET: " << distance);
        ROS_INFO_STREAM("LANELET COVERS CURRENT LOCATION " << isCovered);
        ROS_INFO_STREAM("HEADING ANGLE (atan2(y,x)): " << vehicleHeadingAngle);
        ROS_INFO_STREAM("HEADING ANGLE (degree): " << vehicleHeadingAngle/M_PI * 180.0);
        ROS_INFO_STREAM("LANELET HEADING: " << laneletHeading);
        ROS_INFO_STREAM("ROTATION: (" << vehicleHeading.x << ", " << vehicleHeading.y << ", " << vehicleHeading.z << ")");

        ROS_INFO_STREAM(":::LineStrip:::");        
        ROS_INFO_STREAM("signed distance (bugged): " << signedDistance);
        ROS_INFO_STREAM("distance center line: " << distance_center_line);
        ROS_INFO_STREAM("custom signed distance: " << mySignedDistance);
        ROS_INFO_STREAM("delta: " << std::abs(distance - signedDistance));
        ROS_INFO_STREAM("is Left? " << isLeft);
        ROS_INFO_STREAM("is Right? " << isRight);
        //------------------------------------------------
      }
      else
      {
        // Handle this case!
        ROS_INFO("No suitable lanelet found!");
      }
      poseUpdated = false;
    }

    laneletPublisher.publish(laneletInformationMessage);

    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
