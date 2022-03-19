"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Detects motion in video with sparse optical flow.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 19. 03. 2022
"""

import sys
from pathlib import Path

import numpy as np
import cv2

from src.utils.params import parse_arguments

COMMON_INFO_IDX = 0
POINTS_FOUND_THRESH = 5
POINTS_LOST_THRESH = 5


def load_move_vectors(good_new, good_old):
    vectors = np.empty((2, len(good_new)))

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        vectors[0][i] = a - c
        vectors[1][i] = b - d

    return vectors


def clean_short_vectors(vectors):
    distances = np.linalg.norm(vectors, axis=0)
    point_move_thresh = np.where(distances > 2)
    vectors = vectors[:, point_move_thresh[0]]
    return vectors


def clean_same_vectors(vectors):
    if len(vectors[0]) > 0:
        denominator = vectors.T @ vectors
        norm = (vectors * vectors).sum(0, keepdims=True) ** .5
        similarity_matrix = denominator / norm / norm.T

        indices = []
        for row in range(len(similarity_matrix[0])):
            for col in range(row + 1, len(similarity_matrix[0])):
                indices.append((row, col))

        excluded_indices = []
        for row, col in indices:
            if row in excluded_indices or col in excluded_indices:
                continue
            if similarity_matrix[row, col] > 0.9:
                excluded_indices.append(col)

        vectors = np.delete(vectors, excluded_indices, axis=1)

        # If only one vector remained, all other ones were similar to it and unwanted translation was detected.
        if len(vectors[0]) == 1:
            vectors = [[]]

    return vectors


def calc_move_dist(good_new, good_old):
    vectors = load_move_vectors(good_new, good_old)
    vectors = clean_short_vectors(vectors)
    vectors = clean_same_vectors(vectors)

    if len(vectors[0]) > 0:
        distances = np.linalg.norm(vectors, axis=0)
        average = np.mean(distances)
    else:
        average = 0

    return average


def reduce_flow_in_time(flow, thresh):
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

    print(f"MD -- Loaded video with {frame_length} frames.")

    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Read the first frame.
    ret, frame = video.read()
    if not ret:
        print("MD --- No frame in the video.", file=sys.stderr)
        return np.array([])

    print("MD -- Initial loading of points.")
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    orig_num_points = len(p0)

    for frame_index in range(frame_length):
        # print(f"--- Frame {frame_index}")
        # Read the next frame
        ret, new_frame = video.read()
        if not ret:
            print("MD --- Video length in frames incorrectly computed.", file=sys.stderr)
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
                print(f"MD --- Lost {POINTS_LOST_THRESH} or more points.")
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                orig_num_points = len(p0)
            else:
                p0 = good_new.reshape(-1, 1, 2)
        else:
            print(f"MD --- Have less than {POINTS_FOUND_THRESH} points.")
            old_gray = new_gray.copy()
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            orig_num_points = len(p0)

    return video_flow


class MotionDetector:
    def __init__(self, directory):
        print("Motion Detector (MD) initialized.")
        self.videos = []
        self.frame_length = -1

        print(f"MD - Loading videos from {directory}...")
        for video_path in Path(directory).glob('*.mp4'):
            video = cv2.VideoCapture(str(video_path.resolve()))
            print(f"MD -- Video {video_path} loaded.")

            if self.frame_length == -1:
                self.frame_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            elif self.frame_length != int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1:
                print("MD --- Video lengths do not match!", file=sys.stderr)

            self.videos.append(video)

    def compute_sparse_flows(self):
        print("MD - Computing sparse optical flows...")
        flow = np.zeros((len(self.videos), self.frame_length,))
        for i, video in enumerate(self.videos):
            flow[i] = np.array(get_sparse_flow(video))
            print(f"MD -- Flow from video number {i} computed.")
        flow = flow.sum(axis=0)

        indices, movements = reduce_flow_in_time(flow, len(self.videos) * 20)
        print(f"MD -- Flow from all videos reduced in time.")

        return indices, movements

    def get_indices(self):
        indices, _ = self.compute_sparse_flows()
        return indices


def test():
    args = parse_arguments()

    detector = MotionDetector(args.location)

    for i in detector.get_grid_indices(3):
        print(i)


if __name__ == "__main__":
    test()
