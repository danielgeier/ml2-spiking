import rospy
import sys
from geometry_msgs.msg import Pose, Point, Quaternion


class SetCarBack:
    def __init__(self):
        self.node_name = "setCarBack"
        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.cleanup)

        self.casPose_sub = rospy.Subscriber("/AADC_AudiTT/carPose", Pose, self.setCarBack)
        self.setBack = rospy.Publisher("/AADC_AudiTT/carPoseSet", Pose)


    # def task(self):
    #     self.setBack.publish(Pose(Point(0.00338126594668, -2.74352205099, 0.449979358999),
    #                               Quaternion(-4.21134441427e-07, 5.90477668767e-05, -0.00338638200865, 0.999994264449)))

    def setCarBack (self, pose):
        if (pose.position.z < 0.44):
            print "isch runnergfalle"
            self.setBack.publish(Pose(Point(0.00338126594668, -2.74352205099, 0.449979358999),
                                      Quaternion(-4.21134441427e-07, 5.90477668767e-05, -0.00338638200865,
                                                 0.999994264449)))


    def cleanup(self):
        print "Shutting down vision node."

def main(args):
    try:
        SetCarBack()
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down vision node."

if __name__ == '__main__':
    main(sys.argv)

