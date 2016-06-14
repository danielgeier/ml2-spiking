import cv2
import rospy
import numpy as np
import sys

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from simretina import retina

RECORD_VIDEO = False
NUM_FRAMES = 1000
video = np.zeros((NUM_FRAMES, 50, 100), dtype=np.uint8)
curr_frame = 0

bridge = CvBridge()

eye = retina.init_retina((50, 100))
# 10+ parameters here...
# See http://docs.opencv.org/3.1.0/dc/d54/classcv_1_1bioinspired_1_1Retina.html#ac6d6767e14212b5ebd7c5bbc6477fa7a
eye.setupOPLandIPLParvoChannel(colorMode=False)
eye.setupIPLMagnoChannel(parasolCells_beta=100.,
                         parasolCells_tau=1.,
                         parasolCells_k=10.,
                         amacrinCellsTemporalCutFrequency=2.,
                         # V0CompressionParameter=0.95,
                         # localAdaptintegration_tau=,
                         # localAdaptintegration_k=
                         )


def process_image(ros_image):
    global curr, video, bridge

    try:
        frame = bridge.imgmsg_to_cv2(ros_image)
    except CvBridgeError, e:
        print e
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Collect frames and save them as a .npy file.
    if RECORD_VIDEO:
        video[curr] = frame
        curr += 1
        if curr + 1 >= NUM_FRAMES:
            np.save('driving_video', video)
            sys.exit(0)

    # Get picture from retina
    eye.run(gray)
    parvo_frame = eye.getParvo()
    magno_frame = eye.getMagno()

    # cv2.imshow('parvo', parvo_frame)
    # cv2.imshow('magno', magno_frame)
    # cv2.waitKey(1)

    image_message = bridge.cv2_to_imgmsg(magno_frame, encoding="passthrough")
    aer_pub.publish(image_message)


rospy.init_node("retina")

image_sub = rospy.Subscriber(
    "/AADC_AudiTT/camera_front/image_raw",
    Image,
    process_image,
    queue_size=1,
    buff_size=65536 * 3
)

aer_pub = rospy.Publisher("/dvs/events", Image)

rospy.spin()
