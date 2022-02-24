"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for providing training data in grids.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 13. 02. 2022
"""

import sys
from pathlib import Path
import re

import numpy as np
import cv2

import MotionDetector
from src.utils.params import parse_arguments

COMMON_INFO_IDX = 0


class GridCreator:
    def __init__(self, directory, steps):
        print(f"Loading input files...")
        self.directory = Path(directory)
        self.video_paths = list(Path(directory).glob('*.mp4'))
        self.videos = []
        self.detector = MotionDetector.MotionDetector(directory)

        for video_path in self.video_paths:
            self.videos.append(cv2.VideoCapture(str(video_path.resolve())))

        self.scene = int(re.sub(r"scene(\d+)_cam\d_.*", r"\1", self.video_paths[COMMON_INFO_IDX].stem))
        print(f"- Scene number {self.scene} loaded.")

        self.steps = steps
        self.video_w = int(self.videos[COMMON_INFO_IDX].get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_h = int(self.videos[COMMON_INFO_IDX].get(cv2.CAP_PROP_FRAME_HEIGHT))

    def create_grid(self):
        count = 0
        image_channels = 3

        for indices in self.detector.get_movement_index(self.steps):
            grid = np.empty((self.steps, len(self.videos), self.video_w, self.video_h, image_channels))
            for i in range(len(indices)):
                for video_index in range(len(self.videos)):
                    self.videos[video_index].set(cv2.CAP_PROP_POS_FRAMES, indices[i])
                    ret, frame = self.videos[video_index].read()
                    if not ret:
                        print(f"- Frame {indices[i]:05d} from flow does not exist.", file=sys.stderr)

                    grid[i][video_index] = frame

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
        grid_creator = GridCreator(args.location, args.steps)
    except FileNotFoundError:
        return 1

    grid_creator.detector.compute_sparse_flows()
    grid_creator.save_grids()


if __name__ == "__main__":
    main()
