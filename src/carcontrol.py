import time

import rospy
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from geometry_msgs.msg._Twist import Twist
from std_msgs.msg import Bool, Float64MultiArray

STARTING_POSE = Pose(Point(0.00338126594668, -2.74352205099, 0.449979358999),
                     Quaternion(-4.21134441427e-07, 5.90477668767e-05, -0.00338638200865,
                                0.999994264449))
STARTING_POSE_T = Pose(Point(0.070287314053, 2.4954730954, 0.449979358999),
                     Quaternion(-4.21134441427e-07, 5.90477668767e-05, -0.00338638200865,
                                0.999994264449))

STARTING_ARGS = Vector3(0.0,0.0,0.0)

class CarControlNode:
    """Set out-of-bounds car back to starting position."""

    def __init__(self):
        self.node_name = 'spiking_carcontrol'
        self.distance = None
        self.oldsec = time.time()
        # Disable signals so PyCharm's rerun button works
        rospy.init_node(self.node_name, disable_signals=True)

        self.sub_carPose = rospy.Subscriber('/AADC_AudiTT/carPose', Pose, self.car_of_map)
        self.sub_carDist = rospy.Subscriber('/laneletInformation', Float64MultiArray, self.get_distance)
        self.pub_carModel = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        self.pub_carUpdate = rospy.Publisher('/AADC_AudiTT/carUpdate', Vector3, queue_size=1)
        self.pub_carUpdate = rospy.Publisher('/AADC_AudiTT/carUpdate', Vector3, queue_size=1)
        self.pub = rospy.Publisher('/AADC_AudiTT/carPoseSet', Pose, queue_size=1)
        self.pub_is_set_back = rospy.Publisher('/AADC_AudiTT/isSetBack', Bool, queue_size=1)
        self.reset = False

    def get_distance(self, lanelet_info):
        self.distance = lanelet_info.data[0]
        #self.pub_is_set_back.publish(False)
        self.car_of_lane()

    def car_of_map(self, pose):
        if pose.position.z < 0.44:
            self.reset_car_pose()

    def car_of_lane(self):
        if self.distance <= 0.8:
            self.oldsec = time.time()

        if self.reset:
            self.oldsec= time.time()
            self.reset = False

        newsec = time.time()
        print(self.reset, self.distance, self.oldsec, newsec)
        if newsec - self.oldsec >= 0.5:
            print newsec,'new',self.oldsec

            self.oldsec = time.time()
            self.reset_car_pose()
            self.reset = True

    def reset_car_pose(self):
        print 'isch runnergfalle oder zu lang wegg'
        #change in Starting_pose when using one- or nocrossing
        self.pub.publish(STARTING_POSE_T)
        self.pub_carUpdate.publish(STARTING_ARGS)
        self.pub_is_set_back.publish(True)
        ModelState_x = ModelState()
        ModelState_x.model_name = 'AADC_AudiTT'
        ModelState_x.pose = STARTING_POSE
        ModelState_x.twist = Twist(Vector3(0,0,0),Vector3(0,0,0))
        self.pub_carModel.publish(ModelState_x)

    def reset_car_update(self):
        print 'Speed and Angle reset'
        self.pub_carUpdate.publish(STARTING_ARGS)


if __name__ == '__main__':
    node = CarControlNode()

    while True:
        # Reset car on enter
        raw_input()
        node.reset_car_update()
        node.reset_car_pose()

        # reward = (1 / np.math.sqrt(2 * np.math.pi * varianz )) * np.math.exp(-1 * np.math.pow(distance - mean,2) / 2* varianz)