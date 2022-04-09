"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for providing training data in grids.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 19. 03. 2022
"""

import sys
from pathlib import Path
import re

import numpy as np
import cv2

from src.dataset import MotionDetector
from src.utils.params import parse_arguments

COMMON_INFO_IDX = 0


def get_grid_indices(indices, steps=3):
    grid_indices = []
    for i in range(len(indices) - (steps - 1)):
        for step in range(steps):
            grid_indices.append(indices[i + step])
        yield grid_indices
        grid_indices = []


class GridCreator:
    def __init__(self, directory, steps):
        print("Grid Creator (GC) initialized.")
        self.directory = Path(directory)

        self.steps = steps
        self.detector = MotionDetector.MotionDetector(directory)

        self.videos = []
        self.scene = -1
        self.video_w = -1
        self.video_h = -1

        print(f"GC - Loading videos from {directory}...")
        for video_path in self.directory.glob('*.mp4'):
            video = cv2.VideoCapture(str(video_path.resolve()))
            print(f"GC -- Video {video_path} loaded.")

            if self.scene == -1:
                self.scene = int(re.sub(r"scene(\d+)_cam\d_.*", r"\1", video_path.stem))
            elif self.scene != int(re.sub(r"scene(\d+)_cam\d_.*", r"\1", video_path.stem)):
                print("GC --- Scene numbers do not match!", file=sys.stderr)

            if self.video_w == -1:
                self.video_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            elif self.video_w != int(video.get(cv2.CAP_PROP_FRAME_WIDTH)):
                print("GC --- Video widths do not match!", file=sys.stderr)

            if self.video_h == -1:
                self.video_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            elif self.video_h != int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)):
                print("GC --- Video heights do not match!", file=sys.stderr)

            self.videos.append(video)

        print(f"GC - Scene number {self.scene} loaded.")

    def create_grid(self, indices):
        image_channels = 3

        for indices in get_grid_indices(indices, self.steps):
            grid = np.empty((self.steps, len(self.videos), self.video_w, self.video_h, image_channels))
            for i in range(len(indices)):
                for video_index in range(len(self.videos)):
                    self.videos[video_index].set(cv2.CAP_PROP_POS_FRAMES, indices[i])
                    ret, frame = self.videos[video_index].read()
                    if not ret:
                        print(f"GC -- Frame {indices[i]:05d} from flow does not exist.", file=sys.stderr)

                    grid[i][video_index] = frame

            yield np.hstack(np.hstack(grid))

    def create_grids(self, indices):
        print(f"GC - Exporting grids of images...")

        total = 0

        for count, frames in enumerate(self.create_grid(indices)):
            cv2.imwrite(f"{self.directory}/scene{self.scene:03d}_grid{count:05d}.png", frames)
            if count % 100 == 0:
                print(f"GC -- Grid {count:05d} exported.")
            total = count

        print(f"GC -- Exported {total + 1} grids in total.")


def main():
    args = parse_arguments()

    try:
        grid_creator = GridCreator(args.location, args.steps)
    except FileNotFoundError:
        return 1

    indices = []    # Change to obtaining indices from sparse flow if needed.
    grid_creator.create_grids(indices)


if __name__ == "__main__":
    main()
