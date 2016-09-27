## Synopsis
Machine learning Praktikum 2016 at FZI Karlsruhe
Reinforcement Learning with Spiking Neural Networks
Group 2: Daniel Geier, Simon Di Stefano, Fabian Mack, Lea Steffen


## Getting started
To start on specific lanelet:
    1. run restart_'worldName'.sh (Starts Gazebo)
    2. run camera.py (Gets Camera Sensor Image, and publishes the preprocessed image)
    3. run carcontrol.py (Outside lane controller)
    4. run neuralnet.py (everything else)


## Inventory
./src/camera.py	
		Gets image from camera and preprocesses the image (retina) 
./src/carcontrol.py	
		Sets car back on track
./src/cockpit.py	
		Plots net structure including weights. 
		Plots speed, distance, reward, angle. 
		Offers functionality for debugging
./src/neuralnet.py
		Builds desired net structure, includes learning algorithm, driving behaviour and world state
		Main classes: BaseNetwork, BraitenbergNetwork, DeepNetwork, Learner
		Parameter (plotting & logging): main(argv) n.plot, n.log
		
Custom roads: 		./worlds/onecrossing.world, ./worlds/tcrossing.world
Lanelet specifics: 	./worlds/lanelet_information.cpp, ./worlds/lanelet_random_pos.cpp