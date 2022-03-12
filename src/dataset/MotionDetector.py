"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Detects motion in video with sparse optical flow.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 03. 03. 2022
"""

import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import cv2

from src.utils.params import parse_arguments

COMMON_INFO_IDX = 0
POINTS_FOUND_THRESH = 5
POINTS_LOST_THRESH = 5


def calc_move_dist(good_new, good_old):
    dists = np.empty((len(good_new),))
    m = np.empty((2, len(good_new)))

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        m[0][i] = a - c
        m[1][i] = b - d
        dists[i] = np.linalg.norm([a - c, b - d])

    point_move_thresh = np.where(dists > 2)
    print(f"Thresh: {point_move_thresh}")

    if np.sum(point_move_thresh) > 0:
        average = np.mean(dists[point_move_thresh])

        print(f"Matrix: {m}")
        m = np.squeeze(m[:, point_move_thresh], axis=1)
        print(f"Matrix: {m}")
        d = m.T @ m
        norm = (m * m).sum(0, keepdims=True) ** .5
        similarity_matrix = d / norm / norm.T
        print(f"Similarity mat: {similarity_matrix}")

    else:
        average = 0

    return average


def reduce_flow(flow, thresh):
    indices = [0]
    summed = 0

    for i in range(len(flow)):
        summed += flow[i]
        if summed > thresh:
            if i + 1 >= len(flow):
                indices.append(i)
            else:
                indices.append(i + 1)
            summed = 0

    indices = np.asarray(indices, dtype=np.int64)
    movement = np.add.reduceat(flow, indices)

    return indices, movement


def get_sparse_flow(video):
    """
    Source: https://learnopencv.com/optical-flow-in-opencv/
    """
    frame_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    video_flow = np.zeros((frame_length,))

    print(f"-- Loaded video with {frame_length} frames.")

    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Read the first frame.
    ret, frame = video.read()
    if not ret:
        print("--- No frame in the video.", file=sys.stderr)
        return np.array([])

    print("-- Initial loading of points.")
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    orig_num_points = len(p0)

    for frame_index in range(frame_length):
        # Read the next frame
        ret, new_frame = video.read()
        if not ret:
            print("--- Video length in frames incorrectly computed.", file=sys.stderr)
            break
        new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # Skip the motion detection when less than POINTS_FOUND_THRESH points monitored.
        if orig_num_points >= POINTS_FOUND_THRESH:
            # Calculate Optical Flow
            p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            video_flow[frame_index] = calc_move_dist(good_new, good_old)

            old_gray = new_gray.copy()
            if len(good_new) <= orig_num_points - POINTS_LOST_THRESH:
                print(f"--- Lost {POINTS_LOST_THRESH} or more points.")
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                orig_num_points = len(p0)
            else:
                p0 = good_new.reshape(-1, 1, 2)
        else:
            print(f"--- Have less than {POINTS_FOUND_THRESH} points.")
            old_gray = new_gray.copy()
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            orig_num_points = len(p0)

    return video_flow


class MotionDetector:
    def __init__(self, directory):
        self.video_paths = list(Path(directory).glob('*.mp4'))
        self.movements = []
        self.indices = []

    def compute_sparse_flows(self):
        videos = []
        frame_length = -1

        for video_path in self.video_paths:
            video = cv2.VideoCapture(str(video_path.resolve()))
            frame_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            videos.append(video)

        summed_flow = np.zeros((frame_length,))

        for i, video in enumerate(videos):
            print(f"- Video {i} loaded.")
            flow = np.array(get_sparse_flow(video))
            summed_flow += flow

        self.indices, self.movements = reduce_flow(summed_flow, len(self.video_paths) * 30)

    def get_movement_index(self, steps=2):
        indices = []
        for i in range(len(self.indices) - (steps - 1)):
            for step in range(steps):
                indices.append(self.indices[i + step])
            yield indices
            indices = []


def test():
    args = parse_arguments()

    detector = MotionDetector(args.location)
    detector.compute_sparse_flows()

    for i in detector.get_movement_index(3):
        print(i)


if __name__ == "__main__":
    test()
