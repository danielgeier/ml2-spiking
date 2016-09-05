import threading
import Tkinter as Tk

from vehicle_control.srv import *
import rospy
import numpy as np
import time
import Queue

import cv2
from cv_bridge import CvBridge
from PIL import Image, ImageTk
import sensor_msgs.msg as rosmsg

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

# Mock-Classes for test-purposes


class MockLearner:
    def __init__(self):
        self.world = MockWorld()


class MockState:
    @property
    def distance(self):
        return np.random.rand(1)

    @property
    def angle_vehicle_lane(self):
        return np.random.rand(1)


class MockAgent:

    def __init__(self):
        self.should_learn = True

    @property
    def simduration(self):
        return 20

    @property
    def learner(self):
        return MockLearner()


class MockWorld:
    def calculate_reward(self, actions):
        return np.random.rand(1)

    @property
    def current_state(self):
        return MockState()


class MockNetwork:
    def __init__(self):
        self.weights = np.array([[3.0,2.0],[12.0,2.0]])*128

        self._last_frame = None
        self._bridge = CvBridge()
        self.simduration = 10
        self.learner = MockLearner()

        rospy.init_node('SNN_Cockpit', disable_signals=True)
        self._retina_image_subscriber = rospy.Subscriber('/spiky/retina_image', rosmsg.Image, self._handle_frame,
                                                         queue_size=1)
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

    def decode_actions(self):
        return {'gas': np.random.rand(1), 'brake': np.random.rand(1), 'steering_angle': np.random.rand(1)}


class CockpitViewModel:
    def __init__(self, net, agent):
        self.view = CockpitView(self)

        # Traced variables
        self.should_learn = None
        self.weights_mean_left = None
        self.weights_mean_right = None
        self.use_last = None

        # Important: Start view before initializing the variables
        self.view.start()
        self.net = net
        self.agent = agent

    def initialize_view_model(self):
        self.should_learn = Tk.BooleanVar()
        self.should_learn.set(self.agent.should_learn)
        self.weights_mean_left = Tk.StringVar()
        self.weights_mean_right = Tk.StringVar()
        self.use_last = Tk.BooleanVar()

        self.should_learn.trace("w", self.learn_changed)

    def update(self):
        self.view.update()

    def reset_car_command(self):
        rospy.wait_for_service('reset_car')
        try:
            reset_car_call = rospy.ServiceProxy('reset_car', reset_car)
            mode = 1 if self.use_last.get() else 0
            print "Mode: ", mode
            pose = reset_car_call(mode)
            print pose
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    def reset_weights_command(self):
        self.net.reset_weights()

    def learn_changed(self, *args):
        self.agent.should_learn = not self.agent.should_learn

    def set_weights_command(self, value):
        if value.get() is not None:
            weights = np.array([float(x) for x in value.get().split(',')], dtype=float)
            self.net.set_weights(weights)
        else:
            print 'Keine Gewichte'


class CockpitView(threading.Thread):
    def __init__(self, viewmodel):
        threading.Thread.__init__(self)
        self.viewmodel = viewmodel
        self.root = None
        self.left_weights_mean_label = None
        self.right_weights_mean_label = None
        self.use_last_checkbutton = None
        self.camera_label = None
        self.weights_image_label = None

        # Plot Parents
        self.plot_canvas = None
        self.plot_figure = None

        # Plot axes and Line2Ds
        self.steering_angle_ax = None
        self.reward_ax = None
        self.distance_ax = None
        self.speed_ax = None
        self.angle_vehicle_lane_ax = None

        self.steerin_angle_points = None
        self.reward_points = None
        self.distance_points = None
        self.speed_points = None
        self.angle_vehicle_lane_points = None

        # Step counter for plot
        self.plot_step = 0

        # Plot Data
        self.window_size = 100
        self.steering_angle_data_window = np.zeros(self.window_size)
        self.reward_data_window = np.zeros(self.window_size)
        self.distance_data_window = np.zeros(self.window_size)
        self.speed_data_window = np.zeros(self.window_size)
        self.angle_vehicle_lane_data_window = np.zeros(self.window_size)

    def callback(self):
        self.root.quit()
        self.root.destroy()

    def update(self):
        # Update Camera picture
        self._update_camera_image()
        self._update_weights_image()
        self._update_plots()
        self.plot_step += 1

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
        # image /= 4
        image = cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX) * 255
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.weights_image_label.configure(image=image)
        self.weights_image_label.image = image

    def _update_plots(self):
        actions = self.viewmodel.net.decode_actions()
        world = self.viewmodel.agent.learner.world
        self.steering_angle_data_window[0] = actions['steering_angle']
        self.steering_angle_data_window = np.roll(self.steering_angle_data_window, -1)

        x = np.arange(self.plot_step - self.window_size, self.plot_step) * self.viewmodel.agent.simduration

        self.steerin_angle_points.set_data(x, self.steering_angle_data_window)
        self.steering_angle_ax.set_xlim(np.min(x), np.max(x))
        self.steering_angle_ax.set_ylim(np.min(self.steering_angle_data_window), np.max(self.steering_angle_data_window))

        self.speed_data_window[0] = actions['gas']
        self.speed_data_window = np.roll(self.speed_data_window, -1)
        self.speed_points.set_data(x, self.speed_data_window)
        self.speed_ax.set_ylim(np.min(self.speed_data_window), np.max(self.speed_data_window))

        self.reward_data_window[0] = world.calculate_reward(actions)
        self.reward_data_window = np.roll(self.reward_data_window, -1)
        self.reward_points.set_data(x, self.reward_data_window)
        self.reward_ax.set_ylim(np.min(self.reward_data_window), np.max(self.reward_data_window))

        # car state
        state = world.current_state

        self.distance_data_window[0] = state.distance
        self.distance_data_window = np.roll(self.distance_data_window, -1)
        self.distance_points.set_data(x, self.distance_data_window)
        self.distance_ax.set_ylim(np.min(self.distance_data_window), np.max(self.distance_data_window))

        self.angle_vehicle_lane_data_window[0] = state.angle_vehicle_lane
        self.angle_vehicle_lane_data_window = np.roll(self.angle_vehicle_lane_data_window, -1)
        self.angle_vehicle_lane_points.set_data(x, self.angle_vehicle_lane_data_window)
        self.angle_vehicle_lane_ax.set_ylim(np.min(self.angle_vehicle_lane_data_window),
                                            np.max(self.angle_vehicle_lane_data_window))

        self.plot_canvas.draw()

    def run(self):
        self.root = Tk.Tk()
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.wm_title("Cockpit")

        self.viewmodel.initialize_view_model()
        self.root.protocol("WM_DELETE_WINDOW", self.callback)

        # Create GUI Elements
        leftsiteframe = Tk.Frame(self.root)
        entry_weights_field = Tk.Entry(self.root)
        set_weights_button = Tk.Button(self.root, text="Set Weights", command=lambda: self.viewmodel.set_weights_command(entry_weights_field))
        learn_checkbutton = Tk.Checkbutton(self.root, text="Should Learn?", var=self.viewmodel.should_learn)
        self.use_last_checkbutton = Tk.Checkbutton(self.root, text="Use Last Reset Point?", var=self.viewmodel.use_last)
        reset_car_button = Tk.Button(self.root, text="Reset Car", command=self.viewmodel.reset_car_command)
        constant_weights_text = Tk.Text(self.root, height=1, width=20)
        reset_weights_button = Tk.Button(self.root, text="Reset Weights", command=self.viewmodel.reset_weights_command)
        discount_factor_scale = Tk.Scale(self.root, from_=0, to=100, orient=Tk.HORIZONTAL)

        self.left_weights_mean_label = Tk.Label(self.root, text="left", textvariable=self.viewmodel.weights_mean_left)
        self.right_weights_mean_label = Tk.Label(self.root, text="right", textvariable=self.viewmodel.weights_mean_right)
        self.camera_label = Tk.Label(self.root)
        self.weights_image_label = Tk.Label(self.root)

        # Steering Angle plot
        self.plot_figure = Figure(figsize=(5, 4), dpi=100)
        self.plot_canvas = FigureCanvasTkAgg(self.plot_figure, master=self.root)
        self.plot_canvas.show()

        self.steering_angle_ax = self.plot_figure.add_subplot(511, title="Steering Angle")
        self.speed_ax = self.plot_figure.add_subplot(512, sharex=self.steering_angle_ax, title="Speed")
        self.reward_ax = self.plot_figure.add_subplot(513, sharex=self.steering_angle_ax, title="Reward")
        self.distance_ax = self.plot_figure.add_subplot(514, sharex=self.steering_angle_ax, title="Distance")
        self.angle_vehicle_lane_ax = self.plot_figure.add_subplot(515, sharex=self.steering_angle_ax, title="Angle Vehicle Lane")

        for l in self.steering_angle_ax.get_xticklabels():
            l.set_visible(False)
            l.set_fontsize(0.0)

        for l in self.reward_ax.get_xticklabels():
            l.set_visible(False)
            l.set_fontsize(0.0)

        for l in self.speed_ax.get_xticklabels():
            l.set_visible(False)
            l.set_fontsize(0.0)

        for l in self.distance_ax.get_xticklabels():
            l.set_visible(False)
            l.set_fontsize(0.0)

        self.angle_vehicle_lane_ax.set_xlabel("Simulation Time [ms]")

        self.steerin_angle_points = self.steering_angle_ax.plot(0, 0, '-o', markersize=3)[0]
        self.reward_points = self.reward_ax.plot(0, 0, color='g', marker='o', markersize=3)[0]
        self.distance_points = self.distance_ax.plot(0, 0, 'r-o', markersize=3)[0]
        self.speed_points = self.speed_ax.plot(0, 0, 'c-o', markersize=3)[0]
        self.angle_vehicle_lane_points = self.angle_vehicle_lane_ax.plot(0, 0, 'c-o', markersize=3)[0]

        # Arrange GUI Elements

        leftsiteframe.grid(row=2, column=0)
        learn_checkbutton.grid(row=3, column=0)
        reset_car_button.grid(row=4, column=0)
        self.use_last_checkbutton.grid(row=5, column=0)
        reset_weights_button.grid(row=6, column=0)
        set_weights_button.grid(row=7, column=0)
        entry_weights_field.grid(row=8, column=0)
        # constant_weights_text.grid(row=2, column=1)
        #self.left_weights_mean_label.grid(row=1, column=1)
        #self.right_weights_mean_label.grid(row=2, column=1)
        self.camera_label.grid(row=0,column=0)
        self.weights_image_label.grid(row=1, column=0)
        #discount_factor_scale.grid(row=6, column=1)
        self.plot_canvas.get_tk_widget().grid(row=0, column=1, rowspan=7, sticky='nswe', columnspan=2)

        self.root.mainloop()


if __name__ == '__main__':
    network = MockNetwork()
    agent_ = MockAgent()
    cockpit_view = CockpitViewModel(network, agent_)
    time.sleep(5)

    while True:
        cockpit_view.update()
        time.sleep(0.2)
