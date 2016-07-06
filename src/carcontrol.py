import rospy
import time
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from std_msgs.msg import String

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
        self.sub_carDist = rospy.Subscriber('/AADC_AudiTT/DistanceOneCrossing',String, self.get_distance)
        self.pub_carUpdate = rospy.Publisher('/AADC_AudiTT/carUpdate', Vector3, queue_size=1)
        self.pub = rospy.Publisher('/AADC_AudiTT/carPoseSet', Pose, queue_size=1)

    def get_distance(self, string):
        self.distance = float(string.data)
        self.car_of_lane()

    def car_of_map(self, pose):
        if pose.position.z < 0.44:
            self.reset_car_pose()

    def car_of_lane(self):
        if self.distance <= 1.:
            self.oldsec = time.time()
        newsec = time.time()
        if newsec - self.oldsec >= 2:
            self.reset_car_pose()
            self.oldsec = time.time()

    def reset_car_pose(self):
        print 'isch runnergfalle oder zu lang wegg'
        #change in Starting_pose when using one- or nocrossing
        self.pub.publish(STARTING_POSE_T)
        self.pub_carUpdate.publish(STARTING_ARGS)

    def reset_car_update(self):
        print 'Speed and Angel reset'
        self.pub_carUpdate.publish(STARTING_ARGS)


if __name__ == '__main__':
    node = CarControlNode()

    while True:
        # Reset car on enter
        raw_input()
        node.reset_car_update()
        node.reset_car_pose()

