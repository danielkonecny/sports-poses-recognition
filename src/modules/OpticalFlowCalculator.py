"""Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for calculating optical flow in videos.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 01. 11. 2021
"""

import sys
import os
from argparse import ArgumentParser
from collections import deque

import numpy as np
import cv2


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        'directory',
        type=str,
        help="Path to the directory with videos (without slash at the end).",
    )
    parser.add_argument(
        '-f', '--frame_skip',
        type=int,
        default=7,
        help="Number of frames in between optical flow is calculated."
    )
    return parser.parse_args()


def calc_optical_flow(old_frame, new_frame):
    flow = cv2.optflow.calcOpticalFlowDenseRLOF(old_frame, new_frame, None)

    horizontal_flow = flow[:, :, 0]
    vertical_flow = flow[:, :, 1]

    right_flow = horizontal_flow[horizontal_flow >= 0].sum()
    left_flow = -horizontal_flow[horizontal_flow < 0].sum()
    up_flow = vertical_flow[vertical_flow >= 0].sum()
    down_flow = -vertical_flow[vertical_flow < 0].sum()

    return [up_flow, down_flow, left_flow, right_flow]


class OpticalFlowCalculator:
    def __init__(self, directory: str, frame_skip: int = 7):
        self.directory = directory
        self.files = []
        self.frame_skip = frame_skip

        for file in os.listdir(self.directory):
            if file.endswith('.mp4'):
                self.files.append(file)

    def compute_and_save_flows(self) -> None:
        print(f"Processing all files in {self.directory} directory...")
        for file in self.files:
            flow = self.process_video(file)
            np.save(f'{self.directory}/{file.replace("mp4", "npy")}', flow)
            print(f"- Flow calculated and saved, number of frames: {len(flow)}.")

    def process_video(self, file):
        print(f"Calculating flow of {file} file...")

        frame_queue = deque()

        cap = cv2.VideoCapture(f'{self.directory}/{file}')
        frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Read the first frame
        for i in range(self.frame_skip):
            ret, frame = cap.read()
            if not ret:
                print("- Video shorter than frame skip.", file=sys.stderr)
                return np.array([])
            frame_queue.append(frame)

        video_flow = np.empty((frame_length - self.frame_skip, 4))

        for frame_index in range(frame_length - self.frame_skip):
            old_frame = frame_queue.popleft()

            # Read the next frame
            ret, new_frame = cap.read()
            if not ret:
                print("- Frame length incorrectly computed.", file=sys.stderr)
                break
            frame_queue.append(new_frame)

            video_flow[frame_index] = calc_optical_flow(old_frame, new_frame)

        return video_flow


def main():
    args = parse_arguments()

    optical_flow_calc = OpticalFlowCalculator(args.directory, args.frame_skip)
    optical_flow_calc.compute_and_save_flows()


if __name__ == "__main__":
    main()
