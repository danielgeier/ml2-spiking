import threading
import Tkinter as Tk
from vehicle_control.srv import *
import rospy
import numpy as np
import time
import cv2
from cv_bridge import CvBridge
from PIL import Image, ImageTk
import sensor_msgs.msg as rosmsg


class ModelMock:
    def __init__(self):
        self.weights = np.array([[3.0,2.0],[12.0,2.0]])*128
        self.should_learn = True
        self._last_frame = None
        self._bridge = CvBridge()

        rospy.init_node('SNN_Cockpit', disable_signals=True)
        self._retina_image_subscriber = rospy.Subscriber('/spiky/retina_image', rosmsg.Image, self._handle_frame, queue_size=1)
        rospy.wait_for_message('/spiky/retina_image', rosmsg.Image)

        if self._last_frame is not None:
            shape = np.shape(self._last_frame)
            self._width = shape[0]
            self._height = shape[1]

    def get_weights(self):
        return self.weights

    def set_weights(self, value):
        self.weights = value

    def reset_weights(self):
        pass

    def _handle_frame(self, frame):
        self._last_frame = self._bridge.imgmsg_to_cv2(frame)

    @property
    def last_frame(self):
        return self._last_frame


class CockpitViewModel:
    def __init__(self, net):
        self.view = CockpitView(self)

        # Traced variables
        self.should_learn = None
        self.weights_mean_left = None
        self.weights_mean_right = None

        # Important: Start view before initializing the variables
        self.view.start()
        self.net = net

    def initialize_view_model(self):
        self.should_learn = Tk.BooleanVar()
        self.should_learn.set(self.net.should_learn)
        self.weights_mean_left = Tk.StringVar()
        self.weights_mean_right = Tk.StringVar()

        self.should_learn.trace("w", self.learn_changed)

    def update(self):
        formatstring = "%.4f"
        weights_mean = np.mean(self.net.get_weights(), axis=0)
        self.weights_mean_left.set(formatstring % weights_mean[0])
        self.weights_mean_right.set(formatstring % weights_mean[1])
        self.view.update()

    def reset_car_command(self):
        rospy.wait_for_service('reset_car')
        try:
            reset_car_call = rospy.ServiceProxy('reset_car', reset_car)
            pose = reset_car_call(0)
            print pose
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    def reset_weights_command(self):
        self.net.reset_weights()

    def learn_changed(self, *args):
        self.net.should_learn = not self.net.should_learn


class CockpitView(threading.Thread):
    def __init__(self, viewmodel):
        threading.Thread.__init__(self)
        self.viewmodel = viewmodel
        self.root = None
        self.left_weights_mean_label = None
        self.right_weights_mean_label = None
        self.camera_label = None
        self.weights_image_label = None

    def callback(self):
        self.root.quit()
        self.root.destroy()

    def update(self):
        #Update Camera picture
        self._update_camera_image()
        self._update_weights_image()

    def _update_camera_image(self):
        frame = self.viewmodel.net.last_frame
        image = cv2.resize(frame, (0,0), fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.camera_label.configure(image=image)
        self.camera_label.image = image

    def _update_weights_image(self):
        weights = self.viewmodel.net.get_weights()
        image = cv2.resize(weights, (300,300), interpolation=cv2.INTER_NEAREST)
        image /= 4
        # image = cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX) * 255
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.weights_image_label.configure(image=image)
        self.weights_image_label.image = image

    def run(self):
        self.root = Tk.Tk()
        self.root.columnconfigure(0, weight=1)
        self.root.wm_title("Cockpit")

        self.viewmodel.initialize_view_model()
        self.root.protocol("WM_DELETE_WINDOW", self.callback)

        # Create GUI Elements
        learn_checkbutton = Tk.Checkbutton(self.root, text="Should Learn?", var=self.viewmodel.should_learn)
        reset_car_button = Tk.Button(self.root, text="Reset Car", command=self.viewmodel.reset_car_command)
        constant_weights_text = Tk.Text(self.root, height=1, width=20)
        reset_weights_button = Tk.Button(self.root, text="Reset Weights", command=self.viewmodel.reset_weights_command)
        discount_factor_scale = Tk.Scale(self.root, from_=0, to=100, orient=Tk.HORIZONTAL)

        self.left_weights_mean_label = Tk.Label(self.root, text="left", textvariable=self.viewmodel.weights_mean_left)
        self.right_weights_mean_label = Tk.Label(self.root, text="right", textvariable=self.viewmodel.weights_mean_right)
        self.camera_label = Tk.Label(self.root)
        self.weights_image_label = Tk.Label(self.root)



        # Arrange GUI Elements
        learn_checkbutton.grid(row=0, column=0)
        reset_car_button.grid(row=1, column=0)
        reset_weights_button.grid(row=2, column=0)
        # constant_weights_text.grid(row=2, column=1)
        self.left_weights_mean_label.grid(row=3, column=0)
        self.right_weights_mean_label.grid(row=3, column=1)
        self.camera_label.grid(row=4,column=0)
        self.weights_image_label.grid(row=5, column=0)
        discount_factor_scale.grid(row=6, column=1)

        self.root.mainloop()


if __name__ == '__main__':
    network = ModelMock()
    cockpit_view = CockpitViewModel(network)
    time.sleep(1)

    while True:
        cockpit_view.update()
        time.sleep(0.01)
