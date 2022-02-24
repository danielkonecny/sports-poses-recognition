"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Comparison of different methods for motion detection in video.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 24. 02. 2022
"""

import sys
from pathlib import Path
from collections import deque

import numpy as np
import cv2

from src.utils.params import parse_arguments


def calc_dense_optical_flow(old_frame, new_frame):
    flow = cv2.optflow.calcOpticalFlowDenseRLOF(old_frame, new_frame, None)

    print(f"Dense Flow shape: {flow.shape}")

    horizontal_flow = flow[:, :, 0]
    vertical_flow = flow[:, :, 1]

    right_flow = horizontal_flow[horizontal_flow >= 0].sum()
    left_flow = -horizontal_flow[horizontal_flow < 0].sum()
    up_flow = vertical_flow[vertical_flow >= 0].sum()
    down_flow = -vertical_flow[vertical_flow < 0].sum()

    return [up_flow, down_flow, left_flow, right_flow]


def get_dense_flow(video, frame_skip):
    frame_queue = deque()

    # Read the first couple of frames because of skipping.
    for i in range(frame_skip):
        ret, frame = video.read()
        if not ret:
            print("- Video shorter than frame skip.", file=sys.stderr)
            return np.array([])
        frame_queue.append(frame)

    frame_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_flow = np.empty((frame_length - frame_skip, 4))

    for frame_index in range(frame_length - frame_skip):
        old_frame = frame_queue.popleft()

        # Read the next frame
        ret, new_frame = video.read()
        if not ret:
            print("- Frame length incorrectly computed.", file=sys.stderr)
            break
        frame_queue.append(new_frame)

        video_flow[frame_index] = calc_dense_optical_flow(old_frame, new_frame)

        break

    return video_flow


def lucas_kanade_method(cap):
    frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # Create random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    for frame_index in range(frame_length):
        # Read new frame
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        # Display the demo
        img = cv2.add(frame, mask)
        cv2.imshow("frame", img)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break

        # Update the previous frame and previous points
        print(f"Length {len(good_new)}")
        if len(good_new) < 7:
            # Get new points to follow because many of the previous ones might have got lost.
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        else:
            p0 = good_new.reshape(-1, 1, 2)


def main():
    args = parse_arguments()

    video = cv2.VideoCapture(str(Path(args.location).resolve()))

    lucas_kanade_method(video)

    # video_flow = get_dense_flow(video, args.frame_skip)
    # print(video_flow.shape)
    # print(video_flow)


if __name__ == "__main__":
    main()
