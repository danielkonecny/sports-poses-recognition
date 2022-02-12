"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Detects motion in video with sparse optical flow.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 11. 02. 2022
"""

import sys
from pathlib import Path

import numpy as np
import cv2

from src.utils.params import parse_arguments


COMMON_INFO_IDX = 0


def calc_avg_move_dist(good_new, good_old):
    dist_sum = dist_count = 0

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        dist_sum = np.linalg.norm([a - c, b - d])
        dist_count += 1

    return dist_sum / dist_count


def reduce_flow(flow, thresh):
    indices = [0]
    summed = 0

    for i in range(len(flow)):
        summed += flow[i]
        if summed > thresh:
            indices.append(i + 1)
            summed = 0

    indices = np.asarray(indices, dtype=np.int64)
    movement = np.add.reduceat(flow, indices)
    return indices, movement


def get_sparse_flow(video):
    """
    Source: https://learnopencv.com/optical-flow-in-opencv/
    """
    frame_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    video_flow = np.empty((frame_length,))

    print(f"Frame length: {frame_length}")

    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Read the first frame.
    ret, frame = video.read()
    if not ret:
        print("- No frame in the video.", file=sys.stderr)
        return np.array([])

    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    for frame_index in range(frame_length):
        # Read the next frame
        ret, new_frame = video.read()
        if not ret:
            print("- Video length in frames incorrectly computed.", file=sys.stderr)
            break
        new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        video_flow[frame_index] = calc_avg_move_dist(good_new, good_old)

        old_gray = new_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    return video_flow


def get_movement_index(flow, frame_skip=7, steps=2, move_thresh=80):
    flow_thresh = np.percentile(flow, move_thresh)

    for index in range(len(flow)):
        if index + steps > len(flow):
            break

        # Requires movement above threshold between all steps for all cameras.
        enough_movement = True
        for flow_idx in range(len(flow)):
            for step_idx in range(steps - 1):  # Movement in the last step is not needed.
                if not flow[index + step_idx] > flow_thresh:
                    enough_movement = False

        if enough_movement:
            yield index * frame_skip


def get_frame_from_flow(videos, flows, steps=2, frame_skip=7):
    count = 0
    image_channels = 3
    video_w = int(videos[COMMON_INFO_IDX].get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(videos[COMMON_INFO_IDX].get(cv2.CAP_PROP_FRAME_HEIGHT))

    for index in get_movement_index(flows[0]):
        grid = np.empty((steps, len(videos), video_w, video_h, image_channels))
        frame_index = index
        for step in range(steps):
            for video_index in range(len(videos)):
                videos[video_index].set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = videos[video_index].read()
                if not ret:
                    print(f"- Frame {frame_index:05d} from flow does not exist.", file=sys.stderr)

                grid[step][video_index] = frame

            frame_index += frame_skip

        yield count, np.hstack(np.hstack(grid))
        count += 1


def create_grids_from_flow(videos, flows):
    print(f"\nExporting grids of images...")

    for count, frames in get_frame_from_flow(videos, flows):
        cv2.imwrite(f"grid{count:05d}.png", frames)
        if count % 1 == 0:
            print(f"- Grid {count:05d} exported.")


def get_frame_from_index(videos, indices, steps=2):
    count = 0
    image_channels = 3
    video_w = int(videos[COMMON_INFO_IDX].get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(videos[COMMON_INFO_IDX].get(cv2.CAP_PROP_FRAME_HEIGHT))

    for i in range(len(indices) - (steps - 1)):
        grid = np.empty((steps, len(videos), video_w, video_h, image_channels))

        for step in range(steps):
            frame_index = indices[i + step]

            for video_index in range(len(videos)):
                videos[video_index].set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = videos[video_index].read()
                if not ret:
                    print(f"- Frame {frame_index:05d} from flow does not exist.", file=sys.stderr)

                grid[step][video_index] = frame

        yield count, np.hstack(np.hstack(grid))
        count += 1


def create_grids_from_indices(videos, indices, steps):
    print(f"\nExporting grids of images...")

    for count, frames in get_frame_from_index(videos, indices, steps):
        cv2.imwrite(f"grid{count:05d}.png", frames)
        if count % 1 == 0:
            print(f"- Grid {count:05d} exported.")


def main():
    args = parse_arguments()

    video = cv2.VideoCapture(str(Path(args.location).resolve()))
    flow = get_sparse_flow(video)
    print(f"flow: {flow}")
    indices, movement = reduce_flow(flow, 3)
    print(f"indices: {indices}")
    print(f"movement: {movement}")

    # create_grids_from_flow([video], [movement])
    create_grids_from_indices([video], indices, 6)


if __name__ == "__main__":
    main()
