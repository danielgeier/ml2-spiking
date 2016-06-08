import cv2
import rospy

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from simretina import retina

eye = retina.init_retina((50, 100))
# 10+ parameters here...
# See http://docs.opencv.org/3.1.0/dc/d54/classcv_1_1bioinspired_1_1Retina.html#ac6d6767e14212b5ebd7c5bbc6477fa7a
eye.setupOPLandIPLParvoChannel(colorMode=False, normaliseOutput=False)
eye.setupIPLMagnoChannel()
bridge = CvBridge()


def process_image(ros_image):
    global bridge

    try:
        frame = bridge.imgmsg_to_cv2(ros_image)
    except CvBridgeError, e:
        print e
        return

    # Get picture from retina
    eye.run(frame)
    parvo_frame = eye.getParvo()
    magno_frame = eye.getMagno()

    cv2.imshow('parvo', parvo_frame)
    cv2.imshow('magno', magno_frame)
    cv2.waitKey(1)


rospy.init_node("retina")

image_sub = rospy.Subscriber(
    "/AADC_AudiTT/camera_front/image_raw",
    Image,
    process_image,
    queue_size=1,
    buff_size=65536 * 3
)
