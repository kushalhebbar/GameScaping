GameScaping is a project which incorporates Artificial Intelligence, Data Mining, OpenCV and object detection. This README is a comprehensive guide on the research behind this project as well as a guide to execute the project.

The scenario of driverless vehicles brought popularity in the field of public sector as an on-demand service. People’s expectation increases regularly, and the autonomous driving technique is found to be very reliable. For an example, an autonomous vehicle has no dependency on human for its service, where in case drivers need to get rest after a certain time. The main objective of the autonomous driving needs the object detection technique. When we drive, we constantly pay attention to our environment by avoiding obstacles for our safety. Like introducing sensors, the most important criteria are to detect obstacles. To a seasoned observer, their readings are more specific and profound whereas for autonomous driving they are the representation of their data gets from their navigation and obstacle detector.

Input layer and model:
3 channel RGB features are used as an input. The input for our model can directly be a combination of images and labels but what differentiates our model with other models is the direct streaming input into the network rather than just using defined, trained images and labels. The model identifies the object during stream and learns from them. The longer the model is run the better it gets by gaining knowledge. For the bounding box we use width w and height h in the program. The streaming in our case is actual live gameplay of Grand Theft Auto 5. The game uses Directx11 and runs on the windows platform and requires high GPU powers. Our model should be preferably run on the Nvidia Geforce Titan X as it has been tested on it.
This model is not particular to GTA 5 and can be overlaid on any other game which consists of a decent feature/object definition capability. The reason we decided to run our model on GTA 5 is because of the sandbox feel to it and hence making it very customizable. The game comes with a lot of MODs that allow you to simulate weather, traffic, saturation features. We use OpenCV to grab the screen and that is directly given as the input to the model.
Another scenario for our model is generating and collecting data. This method can be used if in case the data needs to be used in a system where the game cannot be installed. For this model, it is better to collect the data using OpenCV and preferably riding in the game on a scooter or a motorcycle because this gives a 180-degree view and hence the lanes and surroundings are clearly visible. The motorcycle might be too fast for the frames to get generated clearly and hence using a scooter is a much better option.
Since I trained and tested the model on my laptop it has just a CPU and hence set the resolution of the game to 800x600 in windowed mode but if the model is trained and tested on a GPU then the game can be played in its highest resolution. Note that this might generate a large training dataset file due to how rich the frames are. Also note the FPS of the game and decide on the resolution.

Modules of the project:
• Self-driving AI
• Object detection
• Collision detection

Datasets used:
Our model was initially tested on the COCO dataset and upon testing showed high accuracy. Objects such as signals, human beings, cars, trucks, motorcycles, boats, bridges and street lights were recognized with high accuracy. However, it had some inaccuracies where it recognized billboards as a TV and the main menu as a microwave, but it was the most accurate compared to all otherdatasets that we used.
We then tested our model on the Pascal dataset and the results were poor. It recognized most basic objects such as cars and people, but the dataset resulted in a N/A for all other objects.
The third dataset and this test is what we were most optimistic about which was the Kitti dataset. This dataset is produced by Karlsruhe Institute of Technology. These are video sequences from the streets of Karlsruhe. This dataset is good for 3d images where the bounding box and its coordinate axis is converted to a world reference frame by rotation around the y-axis and translation. Kitti datasets are real world first person view footage of scenes and objects around a city. The dataset was able to identify objects such as cars as it had previous exposure to cars, but no other object was identified. The only possible conclusion that I came up with was the difference in pixels between the game and the actual footage and the inaccuracy because of the distortion in the game as there was no GPU.
The following link was used to generate the various datasets: https://github.com/fvisin/dataset_loaders

Please find below the steps to execute the program successfully:
- Object detection and collision detection:
•	Clone or Download the tensorflow repository: https://github.com/tensorflow/models
•	Extract the downloaded repository
•	Follow the steps in this link to install all the required components to get the object detection module working: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
•	After successfully installing all required components based on the instructions navigate to /models/research/object_detection/ and paste the two python files called 
-	vehicle_detector.py
-	grabscreen.py
•	Install the following python packages
-	tensorflow or tensorflow-gpu
-	jupyter
-	matplotlib
-	pillow
-	lxml
-	opencv-python (imported as cv2)
-	numpy
•	Open the game (GTA 5) -> Settings -> Display -> Change to windowed screen and set screen resolution as 800x600 (CPU) maybe higher if executed on GPU
•	Move the game window to the top left corner of the screen
•	Open vehicle_detector.py and run the code
•	Wait for the Opencv window to open (might take a while depending on system processing power)
•	Project is now ready to be tested

- Self-driving AI:

1. Open Python IDLE and run the “gta_vehicle_without_lanes.py” .

2. Open the game (GTA SanAndreas) in window mode and keep it open on the left–top corner. Play the game for a couple of minutes to generate the training data.

3. The training data will be saved on the object-detection folder as “training_data_XX_.npy”

4.After generating the training dataset, for balancing data run the “balance_data_copy.py”. It will show the results of how many left, forward and right moves are done.

5. Install Alexnet with tflearn in windows. Keep the “Alexnet.py” in the object-detection folder.

6. Run the “train_model.py” with the numpy file which was generated earlier in the same directory. The file will generate the metrics for trained values.

7. Open the game again like step 2 and run the file “after_balancing_keys.py”. It will show the prediction values as long as the game will play by itself. Keep the running file minimized and observe the game performance.
     
Note: In the Game first take a scooter and set it to first person view. The game should play in the daylight as it doesn’t produce accurate results if played at night. Set the screen resolution to 800*640.

Google drive link of screenshots and video of gameplay: https://drive.google.com/open?id=1pz7RqAW8oCbnGLjv54Ca0ZZg7tIeVUQZ





