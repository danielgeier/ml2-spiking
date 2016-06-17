import rospy
from geometry_msgs.msg import Pose, Point, Quaternion

STARTING_POSE = Pose(Point(0.00338126594668, -2.74352205099, 0.449979358999),
                     Quaternion(-4.21134441427e-07, 5.90477668767e-05, -0.00338638200865,
                                0.999994264449))


class CarControlNode:
    """Set out-of-bounds car back to starting position."""

    def __init__(self):
        self.node_name = 'spiking_carcontrol'
        # Disable signals so PyCharm's rerun button works
        rospy.init_node(self.node_name, disable_signals=True)

        self.sub = rospy.Subscriber('/AADC_AudiTT/carPose', Pose, self.set_car_back)
        self.pub = rospy.Publisher('/AADC_AudiTT/carPoseSet', Pose, queue_size=1)

    def set_car_back(self, pose):
        if pose.position.z < 0.44:
            self.reset_car_pose()

    def reset_car_pose(self):
        print 'isch runnergfalle'
        self.pub.publish(STARTING_POSE)


if __name__ == '__main__':
    node = CarControlNode()

    while True:
        # Reset car on enter
        raw_input()
        node.reset_car_pose()
