## Synopsis
Machine learning Praktikum 2016 at FZI Karlsruhe</br>
Reinforcement Learning with Spiking Neural Networks</br>
Group 2: Daniel Geier, Simon Di Stefano, Fabian Mack, Lea Steffen</br>


## Getting started
To start on specific lanelet:</br>
    1. run restart_'worldName'.sh (Starts Gazebo)</br>
    2. run camera.py (Gets Camera Sensor Image, and publishes the preprocessed image)</br>
    3. run carcontrol.py (Outside lane controller)</br>
    4. run neuralnet.py (everything else)</br>


## Inventory
./src/camera.py	 </br>
    Gets image from camera and preprocesses the image (retina) </br>
./src/carcontrol.py	</br>
    Sets car back on track</br>
./src/cockpit.py	</br>
    Plots net structure including weights. </br>
    Plots speed, distance, reward, angle. </br>
    Offers functionality for debugging</br>
./src/neuralnet.py</br>
    Builds desired net structure, includes learning algorithm, driving behaviour and world state</br>
    Main classes: BaseNetwork, BraitenbergNetwork, DeepNetwork, Learner</br>
    Parameter (plotting & logging): main(argv) n.plot, n.log</br>
		
Custom roads: 		./worlds/onecrossing.world, ./worlds/tcrossing.world </br>
Lanelet specifics: 	./worlds/lanelet_information.cpp, ./worlds/lanelet_random_pos.cpp </br>