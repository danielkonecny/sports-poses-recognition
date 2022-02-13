"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for providing training data in grids.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 09. 02. 2022
"""

import sys
from pathlib import Path
import re

import numpy as np
import cv2

import DenseMotionDetector
from src.utils.params import parse_arguments

COMMON_INFO_IDX = 0


class GridCreator:
    def __init__(self, directory, move_thresh, steps, frame_skip):
        print(f"Loading input files...")
        self.directory = Path(directory)
        self.video_names = list(Path(directory).glob('*.mp4'))
        self.flow_names = list(Path(directory).glob('*.npy'))
        self.videos = []
        self.flows = []

        for flow_name in self.flow_names:
            self.flows.append(np.load(flow_name))
            if not flow_name.with_suffix(".mp4").is_file():
                print(f"- Missing video file for flow {flow_name}.", file=sys.stderr)
                raise FileNotFoundError

        for video_name in self.video_names:
            self.videos.append(cv2.VideoCapture(str(video_name.resolve())))
            if not video_name.with_suffix(".npy").is_file():
                print(f"- Missing flow file for video {video_name}.", file=sys.stderr)
                raise FileNotFoundError

        for index in range(len(self.videos)):
            print(f"- File {self.video_names[index].stem} with {len(self.flows[index])} flow frames "
                  f"and {self.videos[index].get(cv2.CAP_PROP_FRAME_COUNT)} video frames.")

        self.scene = int(re.sub(r"scene(\d+)_video\d_.*", r"\1", self.video_names[COMMON_INFO_IDX].stem))
        print(f"- Scene number {self.scene} loaded.")

        self.move_thresh = move_thresh
        self.steps = steps
        self.frame_skip = frame_skip
        self.video_w = int(self.videos[COMMON_INFO_IDX].get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_h = int(self.videos[COMMON_INFO_IDX].get(cv2.CAP_PROP_FRAME_HEIGHT))

    def create_grid(self):
        count = 0
        image_channels = 3

        for index in DenseMotionDetector.get_movement_index(self.flows, self.move_thresh, self.steps, self.frame_skip):
            grid = np.empty((self.steps, len(self.videos), self.video_w, self.video_h, image_channels))
            frame_index = index
            for step in range(self.steps):
                for video_index in range(len(self.videos)):
                    self.videos[video_index].set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = self.videos[video_index].read()
                    if not ret:
                        print(f"- Frame {frame_index:05d} from flow does not exist.", file=sys.stderr)

                    grid[step][video_index] = frame

                frame_index += self.frame_skip

            yield count, np.hstack(np.hstack(grid))
            count += 1

    def save_grids(self):
        print(f"\nExporting grids of images...")

        for count, frames in self.create_grid():
            cv2.imwrite(f"{self.directory}/scene{self.scene:03d}_grid{count:05d}.png", frames)
            if count % 100 == 0:
                print(f"- Grid {count:05d} exported.")


def main():
    args = parse_arguments()

    try:
        grid_creator = GridCreator(args.location, args.move_thresh, args.steps, args.frame_skip)
    except FileNotFoundError:
        return 1

    grid_creator.save_grids()


if __name__ == "__main__":
    main()
