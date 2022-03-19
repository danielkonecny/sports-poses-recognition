"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for providing training data for triplet loss.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 19. 03. 2022
"""

import sys
from pathlib import Path
import re

import cv2

import MotionDetector
from src.utils.params import parse_arguments


class DatasetCreator:
    def __init__(self, directory, steps):
        print("Dataset Creator (DC) initialized.")
        self.directory = Path(directory)

        self.steps = steps
        self.detector = MotionDetector.MotionDetector(directory)

        self.videos = []
        self.scene = -1

        print(f"DC - Loading videos from {directory}...")
        for video_path in self.directory.glob('*.mp4'):
            video = cv2.VideoCapture(str(video_path.resolve()))
            print(f"DC -- Video {video_path} loaded.")

            if self.scene == -1:
                self.scene = int(re.sub(r"scene(\d+)_cam\d_.*", r"\1", video_path.stem))
            elif self.scene != int(re.sub(r"scene(\d+)_cam\d_.*", r"\1", video_path.stem)):
                print("DC -- Scene numbers do not match!", file=sys.stderr)

            self.videos.append(video)

        print(f"DC - Scene number {self.scene} loaded.")

    def create_dataset(self):
        print(f"DC - Exporting grids of images...")

        total = 0

        for image_index, index in enumerate(self.detector.get_indices()):
            for video_index, video in enumerate(self.videos):
                video.set(cv2.CAP_PROP_POS_FRAMES, index)
                ret, frame = video.read()
                if not ret:
                    print(f"DC -- Frame {index:05d} from flow does not exist.", file=sys.stderr)

                cv2.imwrite(f"{self.directory}/scene{self.scene:03d}_cam{video_index}_image{image_index:05d}.png",
                            frame)

            if image_index % 10 == 0:
                print(f"DC -- Image {image_index:05d} exported.")

            total = image_index

        print(f"DC -- Exported {total + 1} images in total.")


def main():
    args = parse_arguments()

    try:
        dataset_creator = DatasetCreator(args.location, args.steps)
    except FileNotFoundError:
        return 1

    dataset_creator.create_dataset()


if __name__ == "__main__":
    main()
