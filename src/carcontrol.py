import time

import rospy
from neuralnet import LaneletInformation
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from geometry_msgs.msg._Twist import Twist
from vehicle_control.srv import *
from std_msgs.msg import Bool, Float64MultiArray

STARTING_POSE = Pose(Point(0.00338126594668, -2.74352205099, 0.449979358999),
                     Quaternion(-4.21134441427e-07, 5.90477668767e-05, -0.00338638200865,
                                0.999994264449))
STARTING_POSE_T = Pose(Point(0.070287314053, 2.4954730954, 0.449979358999),
                     Quaternion(-4.21134441427e-07, 5.90477668767e-05, -0.00338638200865,
                                0.999994264449))
RESPAWN_AFTER = 2 #[s]
RESPAWN_Z_THRESHOLD = 0.44
GAZEBO_Z = 0.449979358999

STARTING_ARGS = Vector3(0.0,0.0,0.0)


class CarControlNode:
    """Set out-of-bounds car back to starting position."""
    def __init__(self):
        self.node_name = 'spiking_carcontrol'
        self.distance = None
        self.oldsec = time.time()
        self.oldsec_reset_constantly = time.time()
        # Disable signals so PyCharm's rerun button works
        rospy.init_node(self.node_name, disable_signals=True)
        self.service = rospy.Service('reset_car', reset_car, self.handle_reset_car)

        self.sub_carPose = rospy.Subscriber('/AADC_AudiTT/carPose', Pose, self.car_off_map, queue_size=1)
        self.sub_carDist = rospy.Subscriber('/laneletInformation', Float64MultiArray, self.update_car_state, queue_size=1)
        self.pub_carModel = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        self.pub_carUpdate = rospy.Publisher('/AADC_AudiTT/carUpdate', Vector3, queue_size=1)
        self.pub = rospy.Publisher('/AADC_AudiTT/carPoseSet', Pose, queue_size=1)
        self.pub_is_set_back = rospy.Publisher('/AADC_AudiTT/isSetBack', Bool, queue_size=1)
        self.lanelet_information = None
        self.reset = False


    def random_pose(self):
        rospy.wait_for_service('random_pos_service')
        try:
            random_pos_service_call = rospy.ServiceProxy('random_pos_service', random_pos_service)
            p = random_pos_service_call()
            p = p.output
            starting_position = Point(p[0], p[1], GAZEBO_Z)
            orientation = Quaternion(p[2], p[3], p[4], p[5])
            pose = Pose(starting_position, orientation)
            return {'pose': pose, 'list': p}
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def update_car_state(self, lanelet_info):
        if len(lanelet_info.data) > 0:
            self.distance = lanelet_info.data[0]
            self.lanelet_information = LaneletInformation(lanelet_info.data)
            print self.lanelet_information
            #self.pub_is_set_back.publish(False)
            self.car_off_lane()

    def car_off_map(self, pose):
        self.reset_constantly()
        if pose.position.z < RESPAWN_Z_THRESHOLD:
            self.reset_car_pose()

    def reset_constantly(self):
        if time.time()- self.oldsec_reset_constantly > 120:
            self.reset_car_pose()
            print '--------------------- 2 min ------------------'


    def car_off_lane(self):
        #if self.lanelet_information is None or self.lanelet_information.is_on_lane:
        print self.distance
        if self.distance < 0.5:
            self.oldsec = time.time()

        if self.reset:
            self.oldsec= time.time()
            self.reset = False

        newsec = time.time()
        print(self.reset, self.distance, self.oldsec, newsec)
        if (newsec - self.oldsec) >= RESPAWN_AFTER or self.distance > 5:
            print newsec,'new',self.oldsec

            self.oldsec = time.time()
            print '#######SET_BACK#####'
            print self.distance
            self.reset_car_pose()
            self.reset = True

    def reset_car_pose(self):
        print 'isch runnergfalle oder zu lang wegg'
        res = self.random_pose()
        pose = res['pose']

        #change in Starting_pose when using one- or nocrossing
        self.pub.publish(pose)
        self.pub_carUpdate.publish(STARTING_ARGS)
        self.pub_is_set_back.publish(True)

        modelState = ModelState()
        modelState.model_name = 'AADC_AudiTT'
        modelState.pose = pose
        modelState.twist = Twist(Vector3(0,0,0),Vector3(0,0,0))
        self.pub_carModel.publish(modelState)

        self.oldsec_reset_constantly = time.time()

        return res

    def reset_car_update(self):
        print 'Speed and Angle reset'
        self.pub_carUpdate.publish(STARTING_ARGS)

    def handle_reset_car(self, req):
        res = self.reset_car_pose()
        return reset_carResponse(res['list'])


if __name__ == '__main__':
    node = CarControlNode()
    while True:
        # Reset car on enter
        raw_input()
        node.reset_car_update()
        node.reset_car_pose()

        # reward = (1 / np.math.sqrt(2 * np.math.pi * varianz )) * np.math.exp(-1 * np.math.pow(distance - mean,2) / 2* varianz)