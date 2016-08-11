#include "lanelet_utils.h"

geometry_msgs::Point quaternionRotation(const geometry_msgs::Point& p, const geometry_msgs::Quaternion& q)
{
    //Rotated point
    geometry_msgs::Point r;

    r.x = p.x * (1 - 2*q.y*q.y - 2*q.z*q.z) +  //1 - 2 (y² + z²)
          p.y * 2*(q.x *  q.y - q.w * q.z) +   //2(xy - wz)
          p.z * (q.x * q.z + q.w * q.y);       //2(xz + wy)

    r.y = p.x * (2 * (q.x * q.y + q.w * q.z))  +     //2(xy + wz)
          p.y * (1 - 2 * (q.x * q.x + q.z * q.z) ) + //1 - 2(x²+z²)
          p.z * (2 * (q.y * q.z - q.w * q.x));       //2(yz - wx

    r.z = p.x * (2 * (q.x * q.z - q.w * q.y)) +    //2(xz - wy)
          p.y * (2 * (q.y * q.z + q.w * q.x)) +    //2(yz + wx)
          p.z * (1 - 2 * (q.x * q.x + q.y * q.y)); //1 - 2(x² + y²)

    return r;
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
  p.z = GAZEBO_Z; //adjust according to the Gazebo world and model

  return p;
}

std::string retrieveLaneletFilename(void)
{
    //Retrieve Lanelet filename
    //Filename is registered in the *.launch file of the corresponding world as <param name="laneletFilename" value="..." />
    std::string laneletFilename = "";
    while(ros::ok()) {

      ROS_INFO_STREAM("Trying to retrieve lanelet filename ... ");
      laneletFilename = "";
      bool result = ros::param::get("/AADC_AudiTT/lanelet_filename", laneletFilename);
      ROS_INFO_STREAM("ros::param::get() = " << result);

      if (!laneletFilename.empty()) {
          break;
      }
    }

    return laneletFilename;
}

double determineSide(const LLet::point_with_id_t& p, LLet::lanelet_ptr_t llnet) {
    std::size_t idx = 0;
    std::size_t prevIdx = 0;
    std::size_t nextIdx = 0;
    double angle = 0;

    //Project point
    lanelet_point projected_point = llnet->project(p, &angle, &idx, &prevIdx, &nextIdx);

    //Get neighbors of projected point to calculate lanelet heading vector 'l'
    LLet::strip_ptr_t center_line_strip = boost::get<LLet::SIDE::CENTER>(llnet->bounds());
    std::vector<lanelet_point>& points = center_line_strip->pts();

    LLet::point_with_id_t p1 = points[prevIdx];
    LLet::point_with_id_t p2 = points[nextIdx];

    //lanelet heading vector 'l'
    boost::tuple<double,double> l = boost::make_tuple(boost::get<0>(p2) - boost::get<0>(p1), boost::get<1>(p2) - boost::get<1>(p1));

    //vehicle vector 'v'
    boost::tuple<double,double> v = boost::make_tuple(boost::get<0>(p) - boost::get<0>(p1), boost::get<1>(p) - boost::get<1>(p1));

    //calculate on which side 'v' is located  with respect to 'l': sign(det): -1->left , 0->center, 1->right
    //-> determinant
    double det = boost::get<0>(l)*boost::get<1>(v) - boost::get<1>(l)*boost::get<0>(v);
    double s = det < 0? -1 : (det == 0? 0 : 1); //-1 = left, 0=center, 1=right

    return s;
}

geometry_msgs::Point geom_point(lanelet_point p, bool normalize) {
    return geom_point(boost::get<0>(p), boost::get<1>(p), normalize);
}

geometry_msgs::Point geom_point(double x, double y, bool normalize) {
    geometry_msgs::Point gp;
    gp.x = x;
    gp.y = y;

    if (normalize) {
        double l = std::sqrt(gp.x*gp.x + gp.y*gp.y);
        gp.x /= l;
        gp.y /= l;
    }

    return gp;
}
