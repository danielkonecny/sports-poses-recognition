"""Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for providing training data in grids.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 01. 11. 2021
"""


import os
from argparse import ArgumentParser
import re

import numpy as np
import cv2


COMMON_INFO_IDX = 0


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        'directory',
        type=str,
        help="Path to the directory with videos and flows (without slash at the end).",
    )
    parser.add_argument(
        '-m', '--move_thresh',
        type=int,
        default=80,
        help="Threshold for detecting movement in flow between 0 and 100."
    )
    parser.add_argument(
        '-s', '--steps',
        type=int,
        default=3,
        help="Number of steps in video used."
    )
    parser.add_argument(
        '-f', '--frame_skip',
        type=int,
        default=7,
        help="Number of frames in between optical flow is calculated."
    )
    return parser.parse_args()


class GridCreator:
    def __init__(self, directory, move_thresh, steps, frame_skip):
        print(f"Loading input files...")
        self.directory = directory
        self.files = []
        self.flows = []
        self.videos = []

        for file in os.listdir(self.directory):
            if file.endswith('.npy'):
                self.files.append(file.replace('.npy', ''))
                self.flows.append(np.load(f'{self.directory}/{file}'))
                if not os.path.isfile(f'{self.directory}/{file.replace("npy", "mp4")}'):
                    print(f"- Missing video file for flow {file}.")
            elif file.endswith('.mp4'):
                self.videos.append(cv2.VideoCapture(f'{self.directory}/{file}'))
                if not os.path.isfile(f'{self.directory}/{file.replace("mp4", "npy")}'):
                    print(f"- Missing flow file for video {file}.")

        for index in range(len(self.files)):
            print(f"- File {self.files[index]} with {len(self.flows[index])} flow frames "
                  f"and {self.videos[index].get(cv2.CAP_PROP_FRAME_COUNT)} video frames.")

        self.scene = int(re.sub(r"scene(\d+)_video\d_.*", r"\1", self.files[COMMON_INFO_IDX]))
        print(f"- Scene number {self.scene} loaded.")

        self.move_thresh = move_thresh
        self.steps = steps
        self.frame_skip = frame_skip
        self.video_w = int(self.videos[COMMON_INFO_IDX].get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_h = int(self.videos[COMMON_INFO_IDX].get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_movement_index(self):
        up_flow_thresh = np.empty((len(self.flows)))
        down_flow_thresh = np.empty((len(self.flows)))
        left_flow_thresh = np.empty((len(self.flows)))
        right_flow_thresh = np.empty((len(self.flows)))

        for flow_idx in range(len(self.flows)):
            up_flow_thresh[flow_idx] = np.percentile(self.flows[flow_idx][:, 0], self.move_thresh)
            down_flow_thresh[flow_idx] = np.percentile(self.flows[flow_idx][:, 1], self.move_thresh)
            left_flow_thresh[flow_idx] = np.percentile(self.flows[flow_idx][:, 2], self.move_thresh)
            right_flow_thresh[flow_idx] = np.percentile(self.flows[flow_idx][:, 3], self.move_thresh)

        for index in range(len(self.flows[COMMON_INFO_IDX])):
            if index + self.steps * self.frame_skip > len(self.flows[COMMON_INFO_IDX]):
                break

            # Requires movement above threshold between all steps for all cameras.
            enough_movement = True
            for flow_idx in range(len(self.flows)):
                for step_idx in range(self.steps - 1):  # Movement in the last step is not needed.
                    if not (self.flows[flow_idx][index + step_idx * self.frame_skip][0] > up_flow_thresh[flow_idx] or
                            self.flows[flow_idx][index + step_idx * self.frame_skip][1] > down_flow_thresh[flow_idx] or
                            self.flows[flow_idx][index + step_idx * self.frame_skip][2] > left_flow_thresh[flow_idx] or
                            self.flows[flow_idx][index + step_idx * self.frame_skip][3] > right_flow_thresh[flow_idx]):
                        enough_movement = False

            if enough_movement:
                yield index

    def get_frame(self):
        count = 0
        image_channels = 3

        for index in self.get_movement_index():
            grid = np.empty((self.steps, len(self.videos), self.video_w, self.video_h, image_channels))
            frame_index = index
            for step in range(self.steps):
                for video_index in range(len(self.videos)):
                    self.videos[video_index].set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = self.videos[video_index].read()
                    if not ret:
                        print(f"- Frame {frame_index:05d} from flow does not exist.")

                    grid[step][video_index] = frame

                frame_index += self.frame_skip

            yield count, np.hstack(np.hstack(grid))
            count += 1

    def create_grids(self):
        print(f"\nExporting grids of images...")

        for count, frames in self.get_frame():
            cv2.imwrite(f"{self.directory}/scene{self.scene:03d}_grid{count:05d}.png", frames)
            if count % 100 == 0:
                print(f"- Grid {count:05d} exported.")


def main():
    args = parse_arguments()

    grid_creator = GridCreator(args.directory, args.move_thresh, args.steps, args.frame_skip)
    grid_creator.create_grids()


if __name__ == "__main__":
    main()
