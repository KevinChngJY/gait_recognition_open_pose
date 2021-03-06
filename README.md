
---

## SECTION 1 : PROJECT TITLE (Drafting)
## Gait Recognition

Gait recognition refers to the use of video of human gait, processed by computer vision methods, to recognize or to identify persons based on their body shape and walking styles. This is a passive mode of acquiring biometric data from a distance. It requires an ambient set-up. A RGBD Camera (Intel Realsense) is used to record 2D-image and depth- image of one's walking. With openpose, we extract the skeleton links from 2D-image and restructure to 3D skeleton information using depth-image.

In this tutorial, it demonstrates some ideas to perform person identification/recognition based on the extracted features of 3D skeletons

![image](https://user-images.githubusercontent.com/28354028/119382426-969f5a80-bcf4-11eb-962e-3133b91c57aa.png)

---

## SECTION 2 : Contents

Section 3) Pre-requisites/software/languages
Section 4) Acquire sequence of RGBD and Depth Image fom Intel RealSense <br>
Section 5): Getting started with Intel® RealSense™ SDK 2.0 (v2.44.0) <br>
Section 6) Getting Started with OpenPose <br>
Section 7) 3D de-projection using Depth Image <br>
Section 8) Features Extraction <br>
Section 9) Potential approaches to Gait Recognition <br>
Section 10) Conclusion <br>

---

## SECTION 3 : Pre-requisites/software/languages


---

## SECTION 4 : Acquire sequence of RGBD and Depth Image from Intel RealSense

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
1) Install Intel RealSense SDK 2.0 with Intel RealSense Viewer : https://github.com/IntelRealSense/librealsense/releases/tag/v2.44.0
if your OS is window, you may direct download the exe here : https://github.com/IntelRealSense/librealsense/releases/download/v2.44.0/Intel.RealSense.Viewer.exe
2) After connecting Intel Real Sense camera to you computer via USB, open the installed Intel RealSense Viewer :

![image](https://user-images.githubusercontent.com/28354028/119943254-5cb8a780-bfc5-11eb-809a-7c60f3afc069.png)

After configuring the camera setting, you may start to record your video.

3) Where is your recorded video or file? you may change the recorded folder.

![image](https://user-images.githubusercontent.com/28354028/119943467-aacdab00-bfc5-11eb-81ce-89743ed764a8.png)

Video is recorded in the bag file. later we will use Intel Real Sense library to extract the image (RGB and depth)

---

## SECTION 5 : Getting started with Intel® RealSense™ SDK 2.0 (v2.44.0)

Overview : https://github.com/IntelRealSense/librealsense/releases/tag/v2.44.0

Supported Languages : <br>
1) C++ 11 (GCC 5 / Visual Studio 2015 Update 3) <br>
2) C <br>
3) Python 2.7 / 3.5/ 3.6 / 3.7 / 3.8 / 3.9  :  https://github.com/IntelRealSense/librealsense/tree/development/wrappers/python <br>
4) Node.js : https://github.com/IntelRealSense/librealsense/tree/development/wrappers/nodejs <br>
5) ROS : https://github.com/IntelRealSense/realsense-ros/releases <br>
6) LabVIEW : https://github.com/IntelRealSense/librealsense/tree/development/wrappers/labview<br>
7) .NET : https://github.com/IntelRealSense/librealsense/tree/development/wrappers/csharp<br>
8) Unity : https://github.com/IntelRealSense/librealsense/tree/development/wrappers/unity<br>
9) Matlab : https://github.com/IntelRealSense/librealsense/tree/development/wrappers/matlab<br>
10) OpenNI2 : https://github.com/IntelRealSense/librealsense/tree/development/wrappers/openni2 <br>
11) Unreal Engine 4 : https://github.com/IntelRealSense/librealsense/tree/development/wrappers/unrealengine4 <br>

In this project, we are using python in anaconda, please install pyrealsense2 package in your environment:

```
pip install pyrealsense2
```

More python example:
https://dev.intelrealsense.com/docs/python2

Please refer to the code below to see my example how to extract the images(RGB and Depth) :
[Please take note the annotation to understand them better]

```
# First import the library
import pyrealsense2 as rs
import tqdm
import numpy as np

# your bag file path
bag_path = "path_to_your_bag_file.bag"

# Create a config and get the configure from bag file
config = rs.config()
config.enable_device_from_file(bag_path, repeat_playback=False)

# Create a pipeline
pipleline = rs.pipeline()

# Start the pipeline streaming according to the configuration
profile = pipeline.start(config)

# set real time mode to false
playback = profile.get_device().as_playback()
playback.set_real_time(False)

# get the colorizer filter
# it is the filter generate colour image based on input depth frame
colorizer = rs.colorizer()

# Get the Align Object
align = rs.align(rs.stream.depth)

#Get the duration of video
duration = playback.get_duration().seconds * 1E9 + playback.get_duration().microseconds * 1E3
duration = round(duration/1E6)
elapsed = 0

# Wait until a new set of frames becomes available.
is_present, frames = pipeline.try_wait_for_frames()

# Create a for loop until New frame is not available
while is_present:
            # Get the timestamp from frame
            ts = frames.timestamp
            
            # if there is duplicate processing on the same frame, then skip
            if ts in t:
                # Wait until a new set of frames becomes available.
                is_present, frames = pipeline.try_wait_for_frames()
                continue
            
            # pause the playback for our algorithm to process
            playback.pause()
            
            # align the depth frame to color frame
            frames = align.process(frames)
            
            # update the progress bar
            this = int(playback.get_position()/1E6)-elapsed
            pbar.update(this)
            elapsed += this
            
            # apppend time frame to list of t
            t.append(ts)
            
            # get the colour frame and assign to numpy array
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            
            # get the depth image and color it using color filter, assign to numpy array
            depth_frame = frames.get_depth_frame()
            depth_image = colorizer.colorize(depth_frame)
            depth_image = np.asanyarray(depth_image.get_data())
            
            # assign timeframe, colour image, depth image(original), depth image (colorize) to q 
            # .copy() is critical
            q.append([ts,
                      color_image.copy(), 
                      np.asanyarray(depth_frame.get_data()).copy(), 
                      depth_image.copy()])
                      
            # resumt back the playback
            playback.resume()
            
            # Wait until a new set of frames becomes available
            is_present, frames = pipeline.try_wait_for_frames()
        
        # stop the pipeline streaming
        pipeline.stop()
        
        # update the progress bar and close it
        pbar.update(duration-elapsed)
        pbar.close()

```


We save the following information to variable "q" from bag file:
1) Time Stemp
2) Colour Image (Numpy Array)
3) Depth Image (Original - Numpy Array)
4) Depth Image (After Applying filter - Numpy Array)

---

## SECTION 6 : Getting Started with OpenPose
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
1) Overview of OpenPose of CMU-Perceptual-Computing-Lab ; https://github.com/CMU-Perceptual-Computing-Lab/openpose. <br>
2) Download the source code from https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases <br>

If your laptop don't have GPU, you may download the cpu version, however, it use extremely long time to process the skeleton detection compared to GPU.

3) After downloading the folder, please navigate to the folder "openpose\models" to run the batch script.

![image](https://user-images.githubusercontent.com/28354028/123106659-666de780-d46b-11eb-937a-747eb2ce963e.png)

4) you may refer to my script to extract the information of skeleton using openpose :

```

# First import the library
import cv2
import pyrealsense2 as rs
import os, sys
from tqdm import tqdm

# Import OpenPose
# change the path below accordingly to your downloaded folder
sys.path.append(os.getcwd() + '/openpose/bin/python/openpose/Release');
os.environ['PATH']  = os.environ['PATH'] + ';' + os.getcwd() + '/openpose/bin;'
import pyopenpose as op

# Get OpenPose Object
openpose = op.WrapperPython()

# Configure the object
openpose.configure({'model_folder': './openpose/models', 'display': '0', })

# start openpose
openpose.start()

# Visualization - colour of skeleton
clr = [[0, 255, 0], [0, 255, 255], [0, 0, 255], 
            [0, 255, 0], [0, 255, 255], [0, 0, 255], 
            [255, 0, 0], [255, 255, 0], [255, 0, 255], 
            [255, 0, 0], [255, 255, 0], [255, 0, 255], ]            

# rs bag file name
rsbag = "20210520_222950.bag"

# Variable declaration
pclr = {}
person_library = {}
results[rsbag] = []

# loop over the q - variable "q" is obtaineed from previous script
for i, (ts, image, depth, colorized) in tqdm(enumerate(q[:]),total=len(q[:]),position=0, leave=True, desc=f'Processing openpose'):
        
        # Create pipine for image insert into openpose
        datum = op.Datum()
        
        # read the image
        datum.cvInputData = image
        openpose.emplaceAndPop(op.VectorDatum([datum]))
        
        # if there is no detection from openpose
        if datum.poseKeypoints is None:
            results[rsbag].append((ts, None, None, None))
            continue
        
        skel = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        
        # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
        pose_pairs = op.getPosePartPairs(op.PoseModel.BODY_25)
        pose_pairs = zip(pose_pairs[:-1:2], pose_pairs[1::2])
        pose_pairs = list(pose_pairs)[:14]
        
        # Declare Variable
        plinks, persons, centers = {}, {}, {}
        taken = []
        
        for person in np.arange(datum.poseKeypoints.shape[0]):
            plink = np.zeros((len(pose_pairs), 2, 2)).astype(np.int32)
            # plink[:] = np.nan
            for plink_i, (link0, link1) in enumerate(pose_pairs):
                is_confident = datum.poseKeypoints[person, link0, 2] * datum.poseKeypoints[person, link1, 2]
                if not is_confident:
                    continue
                pt0 = np.around(datum.poseKeypoints[person, link0, :2]).astype(np.int32)
                pt1 = np.around(datum.poseKeypoints[person, link1, :2]).astype(np.int32)
                plink[plink_i,0,:], plink[plink_i,1,:] = pt0, pt1
            plink = rearrange_links(plink)
            x, y = plink[:,:,0].flatten(), plink[:,:,1].flatten()
            x, y = x[(x!=0)&(~np.isnan(x))], y[(y!=0)&(~np.isnan(y))]
            if not x.shape[0] or not y.shape[0]:
                # skip person
                continue
            x0, y0 = x.min(), y.min()
            x1, y1 = x.max(), y.max()
            # perform tracking
            if is_tracking:
                crop = cv2.cvtColor(image[y0:y1+1,x0:x1+1,:].copy(), cv2.COLOR_BGR2GRAY)
                if crop.shape[0] == 0:
                    continue
                ids = perform_tracking(person_library, image, (x0, y0), (x1, y1))
                id = [id for id in ids if id not in taken]
                if len(id) == 0:
                    continue
                id = id[0]
                taken.append(id)
                person_library[id] = crop
            else:
                id = person
            plink_3d, center_3d = de_project(plink, depth, instrinsics)
            centers[id] = center_3d
            persons[id] = plink_3d
            plinks[id] = plink
            if is_plot:
                for clr_i, (pt0, pt1) in enumerate(plink):
                    if 0 in [*pt0, *pt1]:
                        continue
                    pt0, pt1 = tuple(pt0), tuple(pt1)
                    cv2.line(skel, pt0, pt1, clr[clr_i], 3, cv2.LINE_AA)
                    cv2.circle(skel, pt0, 5, [255, 255, 255], -1, cv2.LINE_AA)
                    cv2.circle(skel, pt1, 5, [0, 0, 0], 1, cv2.LINE_AA)
                if id not in pclr:
                    pclr[id] = np.random.uniform(0, 255, 3)
                cv2.rectangle(skel, (x0, y0), (x1, y1), pclr[id], 2)
                cv2.putText(skel, str(id), (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 1, pclr[id], 2)
        if is_plot:
            view = np.hstack([skel, colorized])
            cv2.putText(view, str(i+1), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, np.random.uniform(0, 255, 3), 2)
            cv2.imshow('Stream', view)
            cv2.waitKey(1)
        results[rsbag.split(os.path.sep)[-1]].append((ts, plinks, persons, centers))
    cv2.destroyAllWindows()
    openpose.stop()
    return results
          
                                               




```


---

## SECTION 6 : G3D de-projection using Depth Image

![image](https://user-images.githubusercontent.com/28354028/119386323-f3514400-bcf9-11eb-8ed7-112624125c32.png)

```

def de_project(links, depth, intrinsics, pad=0, max_link_length=1000):
    #  Intrinsic of "Color" / 640x480 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y16}
    #   Width:      	640
    #   Height:     	480
    #   PPX:        	320.625366210938
    #   PPY:        	243.673263549805
    #   Fx:         	381.666015625
    #   Fy:         	381.309906005859
    #   Distortion: 	Inverse Brown Conrady
    #   Coeffs:     	-0.0551249980926514  	0.0641890242695808  	0.000465646124212071  	-0.000292466895189136  	-0.0203048214316368  
    #   FOV (deg):  	79.95 x 64.37
    # intrinsics = rs.intrinsics()
    # intrinsics.width, intrinsics.height = 640, 480
    # intrinsics.ppx, intrinsics.ppy = 320.625366210938, 243.673263549805
    # intrinsics.fx, intrinsics.fy = 381.666015625, 381.309906005859
    # intrinsics.model = rs.distortion.inverse_brown_conrady
    # intrinsics.coeffs = [-0.0551249980926514, 
    #                     0.0641890242695808, 
    #                     0.000465646124212071, 
    #                     -0.000292466895189136, 
    #                     -0.0203048214316368]
    persons = []
    for (w0, h0), (w1, h1) in links:
        if 0 in [w0, h0, w1, h1]:
            persons.append([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])
            continue
        h0_lwr = h0 - pad if h0 - pad >= 0 else 0
        h0_upr = h0 + pad + 1 if h0 + pad + 1 <= depth.shape[0] else depth.shape[0]
        w0_lwr = w0 - pad if w0 - pad >= 0 else 0
        w0_upr = w0 + pad + 1 if w0 + pad + 1 <= depth.shape[1] else depth.shape[1]
        h1_lwr = h1 - pad if h1 - pad >= 0 else 0
        h1_upr = h1 + pad + 1 if h1 + pad + 1 <= depth.shape[0] else depth.shape[0]
        w1_lwr = w1 - pad if w1 - pad >= 0 else 0
        w1_upr = w1 + pad + 1 if w1 + pad + 1 <= depth.shape[1] else depth.shape[1]
        d0 = depth[h0_lwr:h0_upr, w0_lwr:w0_upr].flatten()
        d1 = depth[h1_lwr:h1_upr, w1_lwr:w1_upr].flatten()
        d0, d1 = d0[d0!=0], d1[d1!=0]
        if d0.shape[0] * d1.shape[0] == 0:
            persons.append([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])
            continue
        d0, d1 = d0.min(), d1.min()
        x0, y0, z0 = rs.rs2_deproject_pixel_to_point(intrinsics, [w0, h0], d0)
        x1, y1, z1 = rs.rs2_deproject_pixel_to_point(intrinsics, [w1, h1], d1)
        if np.linalg.norm([x1-x0, y1-y0, z1-z0]) > max_link_length:
            persons.append([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])
            continue
        persons.append([[x0, y0, z0], [x1, y1, z1]])
    center = np.array(persons).reshape([12, 6])
    center = center[~np.any(np.isnan(center), axis=1)]
    center = center.reshape([center.shape[0], 2, 3])
    if center.shape[0] == 0:
        center = np.array([np.nan, np.nan, np.nan])
    else:
        center = np.median(center.mean(axis=1), axis=0)
    return persons, center

```

---

## SECTION 7 : Tracking (Optional)


```

def perform_tracking(person_library, detect_image, top_left_xy, bottom_right_xy, ratio_thresh=0.7):
    (x0, y0), (x1, y1) = top_left_xy, bottom_right_xy
    # gray
    detect_image = cv2.cvtColor(detect_image.copy(), cv2.COLOR_BGR2GRAY)
    # features
    sift = cv2.xfeatures2d.SIFT_create()
    kp2, dc2 = sift.detectAndCompute(detect_image, None)
    scores = []
    for fixed_id, mug_shot in person_library.items():
        kp1, dc1 = sift.detectAndCompute(mug_shot, None)
        # matching
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        # flann
        matches = flann.knnMatch(dc1, dc2, k=2)
        # if dc1 is None or dc2 is None:
        #     continue
        # if dc1.shape[0] <= 1 or dc2.shape[0] <= 1:
        #     continue
        # try:
        #     matches = flann.knnMatch(dc1, dc2, k=2)
        # except cv2.error:
        #     print(dc1, dc2)
        #     continue
        good_points = []
        for match in matches:
            m, n = match
            if m.distance < ratio_thresh * n.distance:
                kp2_idx = m.trainIdx
                x, y = int(kp2[kp2_idx].pt[0]), int(kp2[kp2_idx].pt[1])
                if x0 <= x <= x1 and y0 <= y <= y1:
                    good_points.append((x, y))
        if len(good_points) > 0:
            scores.append((fixed_id, len(good_points)))
    if len(scores) == 0:
        new_id = [len(person_library)]
    else:
        new_id = sorted(scores, key=itemgetter(1), reverse=True)
        new_id = list(map(itemgetter(0), new_id))
    return new_id


```

---

## SECTION 8 : Features Extraction

```

def get_df(datum):
    # convert links to unit vectors
    vectors, timestamp, distance = [], [], []
    for links in datum:
        frame_vector = []
        for link in links:
            t, d, a, b = link[0], link[3::3].mean(), link[1:4], link[4:]
            ab = b - a
            mag = np.linalg.norm(ab)
            ab = ab / mag if mag else np.array([np.nan, np.nan, np.nan])
            frame_vector.append(ab)
        vectors.append(frame_vector)
        timestamp.append(t)
        distance.append(d)
    vectors = np.array(vectors)
    distance = np.array(distance[1:])
    # find rotation axis and rotation angle and interval
    a, b = vectors[:-1], vectors[1:]
    cross = np.cross(a, b)
    mag = np.linalg.norm(cross, axis=2)
    angle = np.arcsin(mag)
    mag = mag.reshape([*mag.shape, 1])
    mag = np.concatenate([mag, mag, mag], axis=2)
    cross /= mag
    r_axis_signal = cross
    r_angle_signal = angle
    interval = np.diff(timestamp)
    # column
    combo = [['RSH', 'RUA', 'RLA', 'LSH', 'LUA', 'LLA', 'RHI', 'RTH', 'RCA', 'LHI', 'LTH', 'LCA'], 
            ['X', 'Y', 'Z']]
    index = pd.MultiIndex.from_product(combo, names=['link', 'axis'])
    # index
    t_delta = []
    ctr = 0
    for i in interval:
        ctr += i/1000
        t_delta.append(ctr)
    t_delta = pd.TimedeltaIndex(t_delta, unit='s')
    # create dataframe
    rot = pd.DataFrame(r_axis_signal.reshape(r_axis_signal.shape[0], 36), columns=index, index=t_delta)
    ang = pd.DataFrame(np.column_stack([distance.reshape(-1, 1), r_angle_signal]), columns=['DST']+combo[0], index=t_delta)
    return rot, ang

```

---

## SECTION 9 : Potential approaches to Gait Recognition

---

## SECTION 10 : Conclusion
