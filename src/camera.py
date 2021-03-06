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

        self.pub_binary = rospy.Publisher("/spiky/binary_image", Image, queue_size=1)
        self.pub_raw = rospy.Publisher("/spiky/raw_image", Image, queue_size=1)
        self.pub_retina = rospy.Publisher("/spiky/retina_image", Image, queue_size=1)

    def process_image(self, ros_image):
        try:
            frame = self.bridge.imgmsg_to_cv2(ros_image)
        except CvBridgeError, e:
            print e
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.eye.run(gray)
        magno_frame = self.eye.getMagno()

        image_message_raw = self.bridge.cv2_to_imgmsg(gray, encoding="passthrough")

        image_message_binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        image_message_binary = image_message_binary[1]
        image_message_binary = self.bridge.cv2_to_imgmsg(image_message_binary, encoding="passthrough")

        image_message_retina = self.bridge.cv2_to_imgmsg(magno_frame, encoding="passthrough")
        self.pub_raw.publish(image_message_raw)
        self.pub_retina.publish(image_message_retina)
        self.pub_binary.publish(image_message_binary)

    def shutdown(self):
        pass


if __name__ == '__main__':
    node = CameraNode()
    rospy.spin()
