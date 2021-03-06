import Tkinter as Tk
import threading
import time
from pyNN import nest
import cv2
import networkx as nx
import numpy as np
import rospy
import sensor_msgs.msg as rosmsg
from PIL import Image, ImageTk
from cv_bridge import CvBridge
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.pyplot import NullLocator
import matplotlib.cm as cm
from vehicle_control.srv import *


import re

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

    @property
    def actor_network(self):
        class MockActorNetwork:
            def __init__(self):
                self.publish_car_actions = True

        return MockActorNetwork()


class MockWorld:
    def calculate_reward(self, actions):
        return np.random.rand(1)

    @property
    def current_state(self):
        return MockState()


class MockNetwork:
    def __init__(self):
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

        self.postsynaptic_learning_neurons = [5033, 5034, 5039, 5040, 5041, 5042, 5043, 5044, 5045, 5046, 5047, 5048]
        self.plastic_connections = [[5027, 5039, 0, 50, 0], [5027, 5040, 0, 50, 1], [5027, 5041, 0, 50, 2],
                                    [5027, 5042, 0, 50, 3], [5027, 5043, 0, 50, 4], [5027, 5044, 0, 50, 5],
                                    [5027, 5045, 0, 50, 6], [5027, 5046, 0, 50, 7], [5027, 5047, 0, 50, 8],
                                    [5027, 5048, 0, 50, 9], [5028, 5039, 0, 50, 0], [5028, 5040, 0, 50, 1],
                                    [5028, 5041, 0, 50, 2], [5028, 5042, 0, 50, 3], [5028, 5043, 0, 50, 4],
                                    [5028, 5044, 0, 50, 5], [5028, 5045, 0, 50, 6], [5028, 5046, 0, 50, 7],
                                    [5028, 5047, 0, 50, 8], [5028, 5048, 0, 50, 9], [5039, 5033, 0, 51, 0],
                                    [5039, 5034, 0, 51, 1], [5040, 5033, 0, 51, 0], [5040, 5034, 0, 51, 1],
                                    [5041, 5033, 0, 51, 0], [5041, 5034, 0, 51, 1], [5042, 5033, 0, 51, 0],
                                    [5042, 5034, 0, 51, 1], [5043, 5033, 0, 51, 0], [5043, 5034, 0, 51, 1],
                                    [5044, 5033, 0, 51, 0], [5044, 5034, 0, 51, 1], [5045, 5033, 0, 51, 0],
                                    [5045, 5034, 0, 51, 1], [5046, 5033, 0, 51, 0], [5046, 5034, 0, 51, 1],
                                    [5047, 5033, 0, 51, 0], [5047, 5034, 0, 51, 1], [5048, 5033, 0, 51, 0],
                                    [5048, 5034, 0, 51, 1]]

        self.spike_events = [{'senders': [5027, 5027, 5027]}, {'senders': [5028, 5028]}]

        self.weights = np.random.rand(len(self.plastic_connections))

    def get_events_spike_detectors(self):
        return self.spike_events

    def get_weights(self):
        return np.random.rand(len(self.plastic_connections))

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

        self.publish_car_actions = Tk.BooleanVar()
        self.publish_car_actions.set(self.agent.actor_network.publish_car_actions)

        self.should_learn.trace("w", self.learn_changed)
        self.publish_car_actions.trace("w", self.publish_car_actions_changed)

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

    def publish_car_actions_changed(self, *args):
        self.agent.actor_network.publish_car_actions = self.publish_car_actions.get()
        print "Publis Car Actions: ", self.publish_car_actions.get()

    def set_weights_command(self, value):
        if value.get() is not None:
            weights = np.array([float(x) for x in (re.split(',|\s+',value.get())) if x != ''], dtype=float)
            self.net.set_weights(weights)
        else:
            print 'Keine Gewichte'


class CockpitView(threading.Thread):

    class BlitDrawing:
        def __init__(self, figure, canvas, ax):
            self.figure = figure
            self.canvas = canvas
            self.ax = ax
            self.background = figure.canvas.copy_from_bbox(ax.bbox)

        def clear(self):
            self.figure.canvas.restore_region(self.background)

        def blit(self):
            self.figure.canvas.blit(self.ax.bbox)

    def __init__(self, viewmodel):
        threading.Thread.__init__(self)
        self.viewmodel = viewmodel
        self.root = None
        self.left_weights_mean_label = None
        self.right_weights_mean_label = None
        self.use_last_checkbutton = None
        self.camera_label = None
        self._update_camera_var = None
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

        self._update_plots_var = None

        # Step counter for plot
        self.plot_step = 0

        # Plot Data
        self.window_size = 100
        self.steering_angle_data_window = np.zeros(self.window_size)
        self.reward_data_window = np.zeros(self.window_size)
        self.distance_data_window = np.zeros(self.window_size)
        self.speed_data_window = np.zeros(self.window_size)
        self.angle_vehicle_lane_data_window = np.zeros(self.window_size)

        # Network Plot
        self.network_plot = None
        self.network_plot_canvas = None
        self.network_plot_ax = None
        self._show_edge_labels_var = None
        self._show_node_labels_var = None
        self._update_graph_var = None
        # Graph and Node positions
        self._nodes_pos = None
        self._G = None


        self.network_blit_drawing = None

    def callback(self):
        self.root.quit()
        self.root.destroy()

    def update(self):
        if self._update_camera_var is None or self._update_plots_var is None or self._update_graph_var is None:
            self.plot_step += 1
            return

        # Update Camera picture
        if self._update_camera_var.get():
            self._update_camera_image()

        if self._update_plots_var.get():
            self._update_plots()

        if self._update_graph_var.get():
            self._update_graph()

        self.plot_step += 1

    def _update_camera_image(self):
        frame = self.viewmodel.net.last_frame
        image = cv2.resize(frame, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.camera_label.configure(image=image)
        self.camera_label.image = image

    def _update_weights_image(self):
        weights = self.viewmodel.net.get_weights()
        image = cv2.resize(weights, (300, 300), interpolation=cv2.INTER_NEAREST)

        # Convert to UC81
        image = np.uint8(cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX) * 255)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Apply Colormap
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
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
        self.steering_angle_ax.set_ylim(np.min(self.steering_angle_data_window),
                                        np.max(self.steering_angle_data_window))

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

    def _prepare_graph(self):
        nodes = self.viewmodel.net.postsynaptic_learning_neurons
        connections = self.viewmodel.net.plastic_connections

        G = nx.DiGraph()

        for node in nodes:
            G.add_node(node)

        i = 0
        # edge_labels = {}
        for conn in connections:
            source = conn[0]
            target = conn[1]
            G.add_edge(source, target)
            # edge_labels[(source, target)] = str(weights[i])
            i += 1

        neuron_layers = {}

        nodes = G.nodes_iter()

        def set_layer(node, layer):
            neighbors = G.neighbors(node)
            for n in neighbors:
                neuron_layers[n] = layer + 1
                set_layer(n, layer + 1)

        for node in nodes:
            in_deg = G.in_degree(node)

            if in_deg == 0:
                neuron_layers[node] = 0
                neighbors = G.neighbors(node)

                for n in neighbors:
                    set_layer(node, 0)

        nodes = np.array([n for n in G.nodes_iter()])
        pos = {}

        max_layer = np.max(neuron_layers.values())
        layers = np.arange(0, max_layer + 1)
        ln = np.array([neuron_layers[x] for x in nodes])

        y_interval = 0 if max_layer == 0 else 1.0 / max_layer

        i = 0
        for layer in layers:
            n = ln == layer
            neurons_in_layer = len(np.nonzero(n)[0])
            x_interval = 1.0 / (neurons_in_layer - 1) if neuron_layers > 1 else 0
            y = i * y_interval
            j = 0

            for node in nodes[n]:
                x = j * x_interval + 0.5
                pos[node] = (x, y)
                j += 1

            i += 1

        self._nodes_pos = pos
        self._G = G

    def _update_graph(self):
        G = self._G
        pos = self._nodes_pos

        self.network_plot_ax.clear()

        connections = self.viewmodel.net.plastic_connections
        weights = self.viewmodel.net.get_weights()
        events_spikes = self.viewmodel.net.get_events_spike_detectors()

        num_spikes_per_neuron = {}
        node_labels = {}
        nodes = [x for x in G.nodes_iter()]

        for n in nodes:
            num_spikes_per_neuron[n] = 0
            node_labels[n] = str(n)

        edge_labels = None
        edges_sign = []

        edges = G.edges_iter()
        edge_labels = {}
        i = 0
        for e in edges:
            edge_labels[e] = '%.1f' % weights[i]
            i += 1

        for e in events_spikes:
            if len(e['senders']) > 0:
                sender = e['senders'][0]
                if sender in nodes:
                    num_spikes_per_neuron[sender] += len(e['senders'])

        weights_abs = np.abs(weights)
        widths = ((weights_abs - np.min(weights_abs)) / (np.max(weights_abs) - np.min(weights_abs)))*4 + 2
        node_colors = [num_spikes_per_neuron[x] for x in G.nodes_iter()]

        self.network_plot_ax.lines = []

        max_abs = np.max(np.abs(weights))

        nx.draw_networkx_nodes(G, pos, ax=self.network_plot_ax,
                               node_color=node_colors,cmap=cm.get_cmap('gist_heat'),vmin=0)

        if self._show_node_labels_var.get():
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=5, ax=self.network_plot_ax, font_color='g'
                                    , font_weight='bold')

        if self._show_edge_labels_var.get() and edge_labels is not None:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=self.network_plot_ax,
                                         label_pos=0.4)

        nx.draw_networkx_edges(G, pos, width=widths, ax=self.network_plot_ax,
                               alpha=0.5,arrows=False,
                               edge_color=weights,
                               edge_vmin=-max_abs,
                               edge_vmax=max_abs,
                               edge_cmap=cm.get_cmap('RdYlGn'))

        for l in self.network_plot_ax.get_xticklabels():
            l.set_visible(False)

        for l in self.network_plot_ax.get_yticklabels():
            l.set_visible(False)

        self.network_plot_ax.xaxis.set_major_locator(NullLocator())
        self.network_plot_ax.yaxis.set_major_locator(NullLocator())

        self.network_plot_canvas.draw()

    def run(self):
        self.root = Tk.Tk()
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.wm_title("Cockpit")

        self.viewmodel.initialize_view_model()

        # Pre-Compute positions of nodes and graph structure
        self._prepare_graph()

        self.root.protocol("WM_DELETE_WINDOW", self.callback)

        # Create GUI Elements
        leftsiteframe = Tk.Frame(self.root)
        entry_weights_field = Tk.Entry(self.root)
        set_weights_button = Tk.Button(self.root, text="Set Weights",
                                       command=lambda: self.viewmodel.set_weights_command(entry_weights_field))
        learn_checkbutton = Tk.Checkbutton(self.root, text="Should Learn?", var=self.viewmodel.should_learn)
        self.use_last_checkbutton = Tk.Checkbutton(self.root, text="Use Last Reset Point?", var=self.viewmodel.use_last)
        reset_car_button = Tk.Button(self.root, text="Reset Car", command=self.viewmodel.reset_car_command)
        reset_weights_button = Tk.Button(self.root, text="Reset Weights", command=self.viewmodel.reset_weights_command)
        publish_action_checkbutton = Tk.Checkbutton(self.root, text="Publish Car Actions?",
                                                    var=self.viewmodel.publish_car_actions)

        self._update_graph_var = Tk.BooleanVar()
        update_graph_checkbutton = Tk.Checkbutton(self.root, text="Update Graph", var=self._update_graph_var)
        self._update_graph_var.set(True)

        self._show_node_labels_var = Tk.BooleanVar()
        show_node_labels_checkbutton = Tk.Checkbutton(self.root, text="Show Node Labels?",
                                                      var=self._show_node_labels_var)
        self._show_node_labels_var.set(False)

        self._show_edge_labels_var = Tk.BooleanVar()
        show_edge_labels_checkbutton = Tk.Checkbutton(self.root, text="Show Edge Labels?",
                                                      var=self._show_edge_labels_var)
        self._show_edge_labels_var.set(False)

        self._update_plots_var = Tk.BooleanVar()
        update_plots_checkbutton = Tk.Checkbutton(self.root, text="Update Plots", var=self._update_plots_var)
        self._update_plots_var.set(True)

        self._update_camera_var = Tk.BooleanVar()
        update_camera_checkbutton = Tk.Checkbutton(self.root, text="Update Camera", var=self._update_camera_var)
        self._update_camera_var.set(True)


        self.left_weights_mean_label = Tk.Label(self.root, text="left", textvariable=self.viewmodel.weights_mean_left)
        self.right_weights_mean_label = Tk.Label(self.root, text="right",
                                                 textvariable=self.viewmodel.weights_mean_right)
        self.camera_label = Tk.Label(self.root)
        self.weights_image_label = Tk.Label(self.root)

        # Car State Plots
        self.plot_figure = Figure(figsize=(5, 4), dpi=100)
        self.plot_canvas = FigureCanvasTkAgg(self.plot_figure, master=self.root)
        self.plot_canvas.show()

        self.steering_angle_ax = self.plot_figure.add_subplot(511, title="Steering Angle")
        self.speed_ax = self.plot_figure.add_subplot(512, sharex=self.steering_angle_ax, title="Speed")
        self.reward_ax = self.plot_figure.add_subplot(513, sharex=self.steering_angle_ax, title="Reward")
        self.distance_ax = self.plot_figure.add_subplot(514, sharex=self.steering_angle_ax, title="Distance")
        self.angle_vehicle_lane_ax = self.plot_figure.add_subplot(515, sharex=self.steering_angle_ax,
                                                                  title="Angle Vehicle Lane")
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

        # Network Topology Plot
        self.network_plot = Figure(figsize=(10, 4), dpi=100)
        self.network_plot_canvas = FigureCanvasTkAgg(self.network_plot, master=self.root)
        self.network_plot_ax = self.network_plot.add_subplot(111, title="Network Topology")

        self.network_plot_canvas.show()

        self.network_blit_drawing = CockpitView.BlitDrawing(self.network_plot, self.network_plot_canvas,
                                                            self.network_plot_ax)
        # Arrange GUI Elements

        leftsiteframe.grid(row=2, column=0)
        learn_checkbutton.grid(row=3, column=0)
        reset_car_button.grid(row=4, column=0)
        self.use_last_checkbutton.grid(row=5, column=0)
        reset_weights_button.grid(row=6, column=0)
        set_weights_button.grid(row=7, column=0)
        entry_weights_field.grid(row=8, column=0)
        update_camera_checkbutton.grid(row=0, column=0)
        self.camera_label.grid(row=1, column=0)
        # self.weights_image_label.grid(row=1, column=0)
        self.plot_canvas.get_tk_widget().grid(row=0, column=1, rowspan=7, sticky='nswe')
        self.network_plot_canvas.get_tk_widget().grid(row=0, column=2, rowspan=7, sticky='nswe')
        update_plots_checkbutton.grid(row=8, column=1)
        update_graph_checkbutton.grid(row=8, column=2)
        show_node_labels_checkbutton.grid(row=9, column=2)
        show_edge_labels_checkbutton.grid(row=10, column=2)
        publish_action_checkbutton.grid(row=9, column=0)

        self.root.mainloop()


if __name__ == '__main__':
    network = MockNetwork()
    agent_ = MockAgent()
    cockpit_view = CockpitViewModel(network, agent_)
    time.sleep(5)

    while True:
        cockpit_view.update()
        time.sleep(0.2)
