import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from simretina import retina


class CameraNode:
    """Generate and publish retina processed images from car camera."""

    def __init__(self):
        self.node_name = 'spiking_camera'
        rospy.init_node(self.node_name)
        self.bridge = CvBridge()
        # See link for full parameter list:
        # http://docs.opencv.org/3.1.0/dc/d54/classcv_1_1bioinspired_1_1Retina.html#ac6d6767e14212b5ebd7c5bbc6477fa7a
        self.eye = retina.init_retina((50, 100))
        self.eye.setupOPLandIPLParvoChannel()
        self.eye.setupIPLMagnoChannel(
            parasolCells_beta=100.,
            parasolCells_tau=1.,
            parasolCells_k=10.,
            amacrinCellsTemporalCutFrequency=2.
        )

        self.sub = rospy.Subscriber(
            "/AADC_AudiTT/camera_front/image_raw",
            Image,
            self.process_image,
            queue_size=1,
            buff_size=65536 * 3
        )

        self.pub = rospy.Publisher("/dvs/events", Image, queue_size=1)

    def process_image(self, ros_image):
        try:
            frame = self.bridge.imgmsg_to_cv2(ros_image)
        except CvBridgeError, e:
            print e
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.eye.run(gray)
        magno_frame = self.eye.getMagno()

        image_message = self.bridge.cv2_to_imgmsg(magno_frame, encoding="passthrough")
        self.pub.publish(image_message)

    def shutdown(self):
        pass


if __name__ == '__main__':
    node = CameraNode()
    rospy.spin()
