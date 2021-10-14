"""Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for providing training data in batches.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 14. 10. 2021
"""


import os
from argparse import ArgumentParser

import numpy as np
import cv2


class BatchProvider:
    def __init__(self, directory):
        self.directory = directory
        self.files = []
        self.flows = []
        self.videos = []

        for file in os.listdir(self.directory):
            if file.endswith('.npy'):
                self.files.append(file.replace('.npy', ''))
                self.flows.append(np.load(f'{self.directory}/{file}'))
                if not os.path.isfile(f'{self.directory}/{file.replace("npy", "mp4")}'):
                    print(f"Missing video file for flow {file}.")
            elif file.endswith('.mp4'):
                self.videos.append(cv2.VideoCapture(f'{self.directory}/{file}'))
                if not os.path.isfile(f'{self.directory}/{file.replace("mp4", "npy")}'):
                    print(f"Missing flow file for video {file}.")

    def loaded_files(self):
        for index in range(len(self.files)):
            print(f"File {self.files[index]} with {len(self.flows[index])} flow frames.")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        'directory',
        type=str,
        help="Path to the directory with videos and flows (without slash at the end).",
    )
    parser.add_argument(
        '-m', '--movement',
        type=int,
        default=1000,   # TODO - determine threshold after tests.
        help="Threshold for detecting movement in flow."
    )
    args = parser.parse_args()

    batch_provider = BatchProvider(args.directory)
    batch_provider.loaded_files()


if __name__ == "__main__":
    main()
