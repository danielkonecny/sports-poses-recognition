"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for calculating optical flow in videos.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 09. 02. 2022
"""

import sys
from pathlib import Path
from collections import deque

import numpy as np
import cv2

from src.utils.params import parse_arguments


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
    def __init__(self, files, frame_skip: int = 7):
        self.files = files
        self.frame_skip = frame_skip

    def compute_and_save_flows(self) -> None:
        print(f"Processing all files in {self.files[0].parent} directory...")
        for file in self.files:
            flow = self.process_video(file)
            np.save(file.with_suffix('.npy'), flow)
            print(f"- Flow calculated and saved, number of frames: {len(flow)}.")

    def process_video(self, file):
        print(f"Calculating flow of {file} file...")

        frame_queue = deque()

        cap = cv2.VideoCapture(str(file.resolve()))
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

    files = list(Path(args.location).glob('*.mp4'))

    optical_flow_calc = OpticalFlowCalculator(files, args.frame_skip)
    optical_flow_calc.compute_and_save_flows()


if __name__ == "__main__":
    main()
