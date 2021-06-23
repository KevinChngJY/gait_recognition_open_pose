# Run this script in shell
# python extract_rsbag.py <source folder> <out file>

import cv2
import pickle
import warnings
import traceback
import numpy as np
import pandas as pd
import pyrealsense2 as rs
from operator import itemgetter
from tqdm import tqdm

# importing openpose
import os, sys
sys.path.append(os.getcwd() + '/openpose/bin/python/openpose/Release');
os.environ['PATH']  = os.environ['PATH'] + ';' + os.getcwd() + '/openpose/bin;'
import pyopenpose as op

warnings.filterwarnings('ignore')


depair = [1, 8, 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 8, 12, 12, 13, 13, 14, 1, 0]
repair = [1, 2, 2, 3, 3, 4, 1, 5, 5, 6, 6, 7, 1, 9, 9, 10, 10, 11, 1, 12, 12, 13, 13, 14]

def rearrange_links(links):
    pts = dict(set(zip(depair, [tuple(lin) for link in links for lin in link])))
    flat_link = [pts[pt] if pt in pts else (np.nan, np.nan) for pt in repair if pt in pts]
    new_links = np.array(list(zip(flat_link[:-1:2], flat_link[1::2])))
    return new_links


def get_frames_from_rsbag(bag_path):
    q, t = [], []
    try:
        config = rs.config()
        config.enable_device_from_file(bag_path, repeat_playback=False)
        pipeline = rs.pipeline()
        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        colorizer = rs.colorizer()
        # align = rs.align(rs.stream.color)
        align = rs.align(rs.stream.depth)
        duration = playback.get_duration().seconds * 1E9 + playback.get_duration().microseconds * 1E3
        duration = round(duration/1E6)
        pbar = tqdm(total=int(duration), position=0, leave=True, desc='Loading images')
        elapsed = 0
        is_present, frames = pipeline.try_wait_for_frames()
        while is_present:
            ts = frames.timestamp
            if ts in t:
                is_present, frames = pipeline.try_wait_for_frames()
                continue
            playback.pause()
            frames = align.process(frames)
            this = int(playback.get_position()/1E6)-elapsed
            pbar.update(this)
            elapsed += this
            t.append(ts)
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            depth_frame = frames.get_depth_frame()
            depth_image = colorizer.colorize(depth_frame)
            depth_image = np.asanyarray(depth_image.get_data())
            # .copy() is critical
            q.append([ts,
                      color_image.copy(), 
                      np.asanyarray(depth_frame.get_data()).copy(), 
                      depth_image.copy()])
            playback.resume()
            is_present, frames = pipeline.try_wait_for_frames()
        pipeline.stop()
        pbar.update(duration-elapsed)
        pbar.close()
    except RuntimeError as e:
        print(str(e))
        traceback.print_exc(file=sys.stdout)
    finally:
        pass
    return q, color_frame.profile.as_video_stream_profile().intrinsics


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


def do_openpose(q, instrinsics, rsbag, results, is_plot, is_tracking):
    #
    openpose = op.WrapperPython()
    openpose.configure({'model_folder': './openpose/models', 'display': '0', })
    openpose.start()
    # 
    rsbag = rsbag.split(os.path.sep)[-1]
    if is_plot:
        clr = [[0, 255, 0], [0, 255, 255], [0, 0, 255], 
            [0, 255, 0], [0, 255, 255], [0, 0, 255], 
            [255, 0, 0], [255, 255, 0], [255, 0, 255], 
            [255, 0, 0], [255, 255, 0], [255, 0, 255], ]
    pclr = {}
    person_library = {}
    for i, (ts, image, depth, colorized) in tqdm(enumerate(q[:]), 
                                                total=len(q[:]),
                                                position=0, leave=True, 
                                                desc=f'Processing openpose'):
        if rsbag not in results:
            results[rsbag.split(os.path.sep)[-1]] = []
        datum = op.Datum()
        datum.cvInputData = image
        openpose.emplaceAndPop(op.VectorDatum([datum]))
        # del openpose
        if datum.poseKeypoints is None:
            results[rsbag.split(os.path.sep)[-1]].append((ts, None, None, None))
            continue
        if is_plot:
            skel = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
        pose_pairs = op.getPosePartPairs(op.PoseModel.BODY_25)
        pose_pairs = zip(pose_pairs[:-1:2], pose_pairs[1::2])
        pose_pairs = list(pose_pairs)[:14]
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


# rsbag_folder = gaits
# rsbag = '20210407_141959.bag'
# out_file = 'plink2.pkl'


if __name__ == '__main__':

    rsbag_folder = sys.argv[1]
    out_file = sys.argv[2]
    is_tracking = False
    is_plot = True

    with open(os.path.join(os.getcwd(), out_file), 'wb') as f:
        pickle.dump({}, f)

    with open(os.path.join(os.getcwd(), out_file), 'rb') as f:
        results = pickle.load(f)
    
    dfs = {}
    ctr = 0
    for rsbag in os.listdir(rsbag_folder):

        ctr += 1
        print(f'\n{ctr}/{len(os.listdir(rsbag_folder))}', 'Processing', rsbag)
        rsbag = os.path.join(os.getcwd(), rsbag_folder, rsbag)

        # read .bag file
        q, i = get_frames_from_rsbag(rsbag)
        print('Total number of frames:', len(q))

        # openpose
        results = do_openpose(q, i, rsbag, results, is_plot, is_tracking)

        # extract 3D links
        data = {}
        for ts, plinks, persons, centers in results[rsbag.split(os.path.sep)[-1]]:
            if persons is None:
                continue
            for person, links in persons.items():
                if person not in data:
                    data[person] = []
                frame_links = []
                for (x0, y0, z0), (x1, y1, z1) in links:
                    frame_links.append(np.array([ts, x0, y0, z0, x1, y1, z1]))
                data[person].append(frame_links)
        
        # create dataframe
        df = {}
        for person, datum in data.items():
            df[person] = get_df(datum)
        dfs[rsbag.split(os.path.sep)[-1]] = df
    
        
    print('\nSuccess:', out_file, '\n')

    with open(out_file, 'wb') as f:
        pickle.dump((results, dfs), f)

    sys.exit(0)
