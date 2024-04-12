# Drift-Line-Lane-Follower
## Overview
**2023 DIFA Daegu Model Autonomous Car Driving Contest: Team DRIFT -- Main driving + setup codes.**   
- Result : **2nd(22.7 sec)**
- front_lidar_avoidance: main driving codes -- contains PD control and OpenCV-based Image processing logic 
- debug/image_color_filter_node.py: color calibration and image pre-processing logic 
## Quick Start
```bash
python3 front_lidar_avoidance.py
```
## Dependencies
* Raspberry Pi 4B+ & Cortex-M3 Controllers
* RGBD Camera & CiLab HW
* Ubuntu 18.04
* Python 2.7
* ROS Melodic
* OpenCV 3.4.6

## Reference
Reference codes for the zzangdol-ai-car project, including implementations of Hector SLAM and Navigation ROS packages,    
are available on GitHub at https://github.com/nsa31/Line-Lane-Follower-Robot_ROS.    
    
These were developed as part of the University of Alberta's CMPUT 412 course on Experimental Robotics at   
https://www.ualberta.ca/computing-science/undergraduate-studies/course-directory/courses/experimental-mobile-robotics.html

### The initial creators of this reference  

* [Nazmus Sakib](https://github.com/nsa31)
* **Vivian**

### Reference Acknowledgement 
* [Programming Robots with ROS](https://github.com/osrf/rosbook/blob/master)







