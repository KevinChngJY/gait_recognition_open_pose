
---

## SECTION 1 : PROJECT TITLE
## Gait Recognition

Gait recognition refers to the use of video of human gait, processed by computer vision methods, to recognize or to identify persons based on their body shape and walking styles. This is a passive mode of acquiring biometric data from a distance. It requires an ambient set-up. A RGBD Camera (Intel Realsense) is used to record 2D-image and depth- image of one's walking. With openpose, we extract the skeleton links from 2D-image and restructure to 3D skeleton information using depth-image.

In this tutorial, it demonstrates some ideas to perform person identification/recognition based on the extracted features of 3D skeletons

![image](https://user-images.githubusercontent.com/28354028/119382426-969f5a80-bcf4-11eb-962e-3133b91c57aa.png)

---

## SECTION 2 : Contents

Section 3) Acquire sequence of RGBD and Depth Image fom Intel RealSense <br>
Section 4) Getting Started with OpenPose <br>
Section 5) 3D de-projection using Depth Image <br>
Section 6) Features Extraction <br>
Section 7) Potential approaches to Gait Recognition <br>
Section 8) Conclusion <br>

---

## SECTION 3 : Acquire sequence of RGBD and Depth Image from Intel RealSense

3 different RGBD cameras you may consider : <br>
(1) Microsoft Kinect : https://www.microsoft.com/en-us/p/azure-kinect-dk/8pp5vxmd9nhq?activetab=pivot%3aoverviewtab <br>
(2) Intel® RealSense™ Depth Camera D435 : https://www.intelrealsense.com/compare-depth-cameras/ <br>
(3) ZED Stereo Camera (StereoLabs) : https://www.stereolabs.com/zed/ <br>

They have well-support package for preprocessing, colour-allignment, etc. However, you may get your own depth camera to follow this tutorial. <br>
For this tutorial, i use Intel RealSense d435i. 

Below is the configuration of my camera when recording my dataset:

| Index  |  Subject | Remarks |
| :------------ | :-----------------------|:----------------|
| 1 | Camera | intel realsense d435i |
| 2 | Recordings | Sequence of RGB Image and Depth Image (Video) |
| 3 | Resolution | 640 x 480 |
| 4 | Frame Rate | 15s |
| 5 | Depth and Colour Space | Z16(Depth) and RGB(Colour) |
| 6 | Processing (Camera In-built) | Enable Auto-exposure |
| 7 | Walking Direction | Walking towards Camera |
| 8 | The Distance of person away from Camera | 20-3 meters |
| 9 | Number of Persons/labels | 5 |
| 10 | Number of Observations | 10 Observations/recordings per each person |
| 11 | In/Outdoor | Outdoor |

#### you may refer to my steps to record the data using Intel Realsense d435i:


---

## SECTION 4 : Getting Started with OpenPose
### Guide to OpenPose for Real-time Human Pose Estimation

In this section, we will walk you through how can we get the skeleton links of human using OpenPose.

#### Introduction to Openpose

OpenPose is a Real-time multiple-person detection library, and it’s the first time that any library has shown the capability of jointly detecting human body, face, and foot keypoints. Thanks to Gines Hidalgo, Zhe Cao, Tomas Simon, Shih-En Wei, Hanbyul Joo, and Yaser Sheikh for making this project successful and this library is very much dependent on the CMU Panoptic Studio dataset. OpenPose is written in C++ and Caffe. Today we are going to see a very popular library with almost a 19.8k star and 6k fork on Github: OpenPose with a small implementation in python, the authors have created many builds for different operating systems and languages. You can try it in your local machine with GPU or without GPU, with Linux or without Linux.

There are many features of OpenPose library let’s see some of the most remarkable ones:
* Real-time 2D multi-person keypoint detections.
* Real-time 3D single-person keypoint detections.
* Included a Calibration toolbox for estimation of distortion, intrinsic, and extrinsic camera parameters.
* Single-person tracking for speeding up the detection and visual smoothing.

#### Steps to use Openpose

---

## SECTION 5 : G3D de-projection using Depth Image

![image](https://user-images.githubusercontent.com/28354028/119386323-f3514400-bcf9-11eb-8ed7-112624125c32.png)

---

## SECTION 6 : Features Extraction

---

## SECTION 7 : Potential approaches to Gait Recognition

---

## SECTION 8 : Conclusion
