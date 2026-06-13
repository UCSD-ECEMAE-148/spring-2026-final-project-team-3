# Golf Ball Collector - Spring 2026 Final Project (Team 3)

### *UCSD ECEMAE-148 | Spring 2026*

---

## Demo Video

*(Insert demo video link here)*

---

## Team Members

* Connor Phon
* Dominick Cardella
* Randy Salazar
* Parth Bhagwat

---

## Project Overview

This is an autonomous golfball collector. It uses computer vision based golfball detection using an Oak D Lite camera. It uses PID to control steering based on the detection data. It also uses LIDAR for obstacle avoidance. This runs on a Raspberry Pi over SSH and can be used with a Logitech controller.

### Key Features

- Golfball detection using a YOLO model, running inference directly on Oak D Lite instead of the Pi for lower latency
- PID control for steering inputs based on golfball detection
- Optional video stream to localhost showing detected golfballs and current target
- LIDAR obstacle detection and avoidance
- Controller input for autonomous and manual drive modes (Using Logitech F710, press 'Y' to toggle drive mode and use left/right joysticks for manual driving)
- Set a maximum distance from starting point (uses VESC odometry to determine distance)

---

## Hardware

| Component | Details |
|--------------|-----------------|
| Platform | Acrylic RC car chassis |
| Compute | Raspberry Pi |
| Camera | Oak D Lite |
| LIDAR | SICK lidar |
| Controller | Logitech F710 | 

--- 

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




