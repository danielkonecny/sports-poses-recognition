"""Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for providing training data in batches.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 16. 10. 2021
"""


import os
from argparse import ArgumentParser
import re

import numpy as np
import cv2


def get_movement_index(flow, percentile_thresh=95):
    up_flow_thresh = np.percentile(flow[:, 0], percentile_thresh)
    down_flow_thresh = np.percentile(flow[:, 1], percentile_thresh)
    left_flow_thresh = np.percentile(flow[:, 2], percentile_thresh)
    right_flow_thresh = np.percentile(flow[:, 3], percentile_thresh)
    for index in range(len(flow)):
        if flow[index][0] > up_flow_thresh or \
                flow[index][1] > down_flow_thresh or \
                flow[index][2] > left_flow_thresh or \
                flow[index][3] > right_flow_thresh:
            yield index


class BatchCreator:
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

        self.scene = int(re.sub(r"scene(\d+)_video\d_.*", r"\1", self.files[0]))
        print(f"- Scene number {self.scene} loaded.")

        self.move_thresh = move_thresh
        self.steps = steps
        self.frame_skip = frame_skip
        self.video_w = int(self.videos[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_h = int(self.videos[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame(self):
        count = 0
        movement = get_movement_index(self.flows[1], self.move_thresh)

        for index in movement:
            batch = np.empty((self.steps, len(self.videos), self.video_w, self.video_h, 3))
            frame_index = index
            for step in range(self.steps):
                for video_index in range(len(self.videos)):
                    self.videos[video_index].set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = self.videos[video_index].read()
                    if not ret:
                        print(f"- Frame {frame_index:05d} from flow does not exist.")

                    batch[step][video_index] = frame

                frame_index += self.frame_skip

            yield count, np.hstack(np.hstack(batch))
            count += 1

    def create_batches(self):
        print(f"Exporting batches of images...")
        image_provider = self.get_frame()

        for count, frames in image_provider:
            cv2.imwrite(f"{self.directory}/scene{self.scene:03d}_batch{count:05d}.png", frames)
            if count % 100 == 0:
                print(f"- Batch {count:05d} exported.")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        'directory',
        type=str,
        help="Path to the directory with videos and flows (without slash at the end).",
    )
    parser.add_argument(
        '-m', '--move_thresh',
        type=int,
        default=95,
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
    args = parser.parse_args()

    batch_creator = BatchCreator(args.directory, args.move_thresh, args.steps, args.frame_skip)
    batch_creator.create_batches()


if __name__ == "__main__":
    main()
