"""Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for synchronization of multiple videos of the same scene.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 06. 10. 2021
"""

from argparse import ArgumentParser
from collections import deque

import numpy as np
import cv2


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--video_path", default="../dip-data/in/VID_20211006_194430_ffmpeged.mp4", help="Path to the video",
    )

    args = parser.parse_args()
    video_path = args.video_path
    frame_skip = 7
    frame_queue = deque()

    cap = cv2.VideoCapture(video_path)
    frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read the first frame
    for i in range(frame_skip):
        ret, frame = cap.read()
        if not ret:
            print("Video shorter than frame skip.")
            return
        frame_queue.append(frame)

    video_flow = np.empty((frame_length - frame_skip, 4))

    for frame_index in range(frame_length - frame_skip):
        old_frame = frame_queue.popleft()
        # Read the next frame
        ret, new_frame = cap.read()
        if not ret:
            print("Frame length incorrectly computed.")
            break
        frame_queue.append(new_frame)

        # Calculate Optical Flow
        flow = cv2.optflow.calcOpticalFlowDenseRLOF(old_frame, new_frame, None)

        horizontal_flow = flow[:, :, 0]
        vertical_flow = flow[:, :, 1]

        right_flow = horizontal_flow[horizontal_flow >= 0].sum()
        left_flow = -horizontal_flow[horizontal_flow < 0].sum()
        up_flow = vertical_flow[vertical_flow >= 0].sum()
        down_flow = -vertical_flow[vertical_flow < 0].sum()

        video_flow[frame_index] = [up_flow, down_flow, left_flow, right_flow]

    print(video_flow)
    print(video_flow.shape)

    np.save('out/flow_20211009_2.npy', video_flow)


if __name__ == "__main__":
    main()
