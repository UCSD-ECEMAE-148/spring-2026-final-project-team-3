# Golf Ball Collector - Spring 2026 Final Project (Team 3)

### *UCSD ECEMAE-148 | Spring 2026*

---

![Alt text](./car.jpg)

---

## Team Members

| Name | Department |
|--------------|-----------------|
| Connor Phon | ECE |
| Dominick Cardella | ECE |
| Parth Bhagwat | MAE |
| Randy Salazar | MAE |

---

## Abstract

This is an autonomous golfball collector. It uses computer vision based golfball detection using an Oak D Lite camera. It uses PID to control steering based on the detection data. It also uses LIDAR for obstacle avoidance. This runs on a Raspberry Pi over SSH and can be used with a Logitech controller.

### Key Features

- Golfball detection using a YOLO model, running inference directly on Oak D Lite instead of the Pi for lower latency
- PID control for steering inputs based on golfball detection
- Optional video stream to localhost showing detected golfballs and current target
- LIDAR obstacle detection and avoidance
- Controller input for autonomous and manual drive modes (Using Logitech F710, press 'Y' to toggle drive mode and use left/right joysticks for manual driving)
- Set a maximum distance from starting point (uses VESC odometry to determine distance)

---

### What we Promised

Golfball collection mechanism

Camera-based computer vision

Lidar based obstacle detection

ROS2 integration between perception and actuation

### Stretch Goals

SLAM for mapping out the surrounding area using LIDAR. 

Tilting LIDAR mount to get a 3D topographical view of the area.

## Hardware

| Component | Details |
|--------------|-----------------|
| Platform | Lasercut Acrylic |
| Compute | Raspberry Pi |
| Camera | Oak D Lite |
| LIDAR | SICK lidar |
| Controller | Logitech F710 | 

--- 

## Software

Our code is based on the UCSDRobocar Docker environment: 
 https://github.com/ucsd-ecemae-148/ucsd_robocar_hub2/pkgs/container/ucsd_robocar. 

The custom code we wrote for this project is mainly within the `golfball` package. 
Within `src/golfball/golfball` we have the following nodes: 

- `ball_detection_node.py` 
    
    This starts up the camera and runs the golfball detection model on the Oak D Lite. It uses the DepthAI library to run inference directly on the Oak D Lite rather than the Pi CPU. It publishes the detection data to the `/detected_balls` topic.

- `stream.py`

    This gets the video input and golfball tracking data and displays it on `localhost:5000`. This makes debugging easier.

- `lidar_node.py`

    This reads data from the SICK lidar to find distances to the nearest objects in our surroundings. It publishes suggested steering inputs based on the lidar data. 

- `controller_node.py`

    Pretty much just reads controller inputs and publishes to `/controller_inputs` topic

- `odom_sub_node.py`

    Uses VESC odometry to determine the pose of the car. It can determine the car's distance from its starting location and its angular position. Then it suggests that the car rotates back toward the origin.

- `drive_node.py`

    The node that actually drives the car. It receives data from all the above nodes (except for the stream node) and determines steering and throttle inputs. It handles the servo PID for the golfball collection. Additionally it handles the priorities from the other sensor inputs to figure out where to steer.


## How to Run

### 1. Download the UCSDRoboCar Docker Image

Here's a link: https://github.com/ucsd-ecemae-148/ucsd_robocar_hub2/pkgs/container/ucsd_robocar

### 2. Clone the Repository

```
git clone git@github.com:UCSD-ECEMAE-148/spring-2026-final-project-team-3.git
```

### 3. Setting up the ROS environment

Source the workspace

```
cd /home/projects/ros2_ws
source /opt/ros/foxy/setup.bash
source install/setup.bash
```

Build the workspace

```
colcon build
```

Also make sure to set ROS_DOMAIN_ID to (in our case 3 since we're team 3)

```
export ROS_DOMAIN_ID=3
```

### 4. Starting the ROS nodes

Now launch golfball.launch.py

```
ros2 launch golfball golfball.launch.py
```

Now the car should be in manual drive mode. Use the joysticks on the Logitech controller to drive the car. Or press 'Y' to switch to autonomous driving.




