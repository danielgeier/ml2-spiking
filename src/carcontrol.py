import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3

STARTING_POSE = Pose(Point(0.00338126594668, -2.74352205099, 0.449979358999),
                     Quaternion(-4.21134441427e-07, 5.90477668767e-05, -0.00338638200865,
                                0.999994264449))
STARTING_ARGS = Vector3(0.0,0.0,0.0)

class CarControlNode:
    """Set out-of-bounds car back to starting position."""

    def __init__(self):
        self.node_name = 'spiking_carcontrol'
        # Disable signals so PyCharm's rerun button works
        rospy.init_node(self.node_name, disable_signals=True)

        self.sub_carPose = rospy.Subscriber('/AADC_AudiTT/carPose', Pose, self.set_car_back)
        self.pub_carUpdate = rospy.Publisher('/AADC_AudiTT/carUpdate', Vector3, queue_size=1)
        self.pub = rospy.Publisher('/AADC_AudiTT/carPoseSet', Pose, queue_size=1)

    def set_car_back(self, pose):
        if pose.position.z < 0.44:
            self.reset_car_pose()

    def reset_car_pose(self):
        print 'isch runnergfalle'
        self.pub.publish(STARTING_POSE)
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

