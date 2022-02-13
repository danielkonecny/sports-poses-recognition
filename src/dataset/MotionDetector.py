"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Detects motion in video with sparse optical flow.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 13. 02. 2022
"""

import sys
from pathlib import Path

import numpy as np
import cv2

from src.utils.params import parse_arguments

COMMON_INFO_IDX = 0


def calc_avg_move_dist(good_new, good_old):
    dist_sum = dist_count = 0

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        dist_sum = np.linalg.norm([a - c, b - d])
        dist_count += 1

    return dist_sum / dist_count


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
    video_flow = np.empty((frame_length,))

    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Read the first frame.
    ret, frame = video.read()
    if not ret:
        print("- No frame in the video.", file=sys.stderr)
        return np.array([])

    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    for frame_index in range(frame_length):
        # Read the next frame
        ret, new_frame = video.read()
        if not ret:
            print("- Video length in frames incorrectly computed.", file=sys.stderr)
            break
        new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        video_flow[frame_index] = calc_avg_move_dist(good_new, good_old)

        old_gray = new_gray.copy()
        if frame_index % 100 == 0:
            # Get new points to follow because many of the previous ones might have got lost.
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        else:
            p0 = good_new.reshape(-1, 1, 2)

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

        for video in videos:
            flow = np.array(get_sparse_flow(video))
            summed_flow += flow

        self.indices, self.movements = reduce_flow(summed_flow, len(self.video_paths) * 4)

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