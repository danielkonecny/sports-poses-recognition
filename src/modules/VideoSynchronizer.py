"""Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for synchronization of multiple videos of the same scene.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 12. 10. 2021
"""

import os
from argparse import ArgumentParser

import numpy as np
from scipy.stats import pearsonr
import cv2

import OpticalFlowCalculator


def get_timestamp_from_seconds(seconds):
    milliseconds = int((seconds % 1) * 1000)
    hours = 0
    minutes = 0
    if seconds > 3600:
        hours = int(seconds // 3600)
        seconds %= 3600
    if seconds > 60:
        minutes = int(seconds // 60)
        seconds %= 60
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{milliseconds:03d}."


class VideoSynchronizer:
    def __init__(self, directory, overlay=1000):
        self.directory = directory
        self.overlay = overlay
        self.videos = []
        self.flows = []
        self.differences = [0]
        self.fps = []

    def load_flows(self):
        print("Loading flows.")

        for file in os.listdir(self.directory):
            if file.endswith('.npy'):
                self.flows.append(np.load(f'{self.directory}/{file}'))
                self.videos.append(file.replace("npy", "mp4"))

        print(f"Flows loaded: {len(self.flows)}.")

    def calculate_flows(self):
        print("Calculating flows.")

        optical_flow_calc = OpticalFlowCalculator.OpticalFlowCalculator(self.directory)

        for file in os.listdir(self.directory):
            if file.endswith('.mp4'):
                flow = optical_flow_calc.process_video(file)
                self.flows.append(flow)
                self.videos.append(file)

        print(f"Flows calculated: {len(self.flows)}.")

    def check_length(self):
        print("Checking length of flows.")

        flow1 = self.flows[0]
        flow2 = self.flows[1]

        video1 = self.videos[0]
        video2 = self.videos[1]

        if len(flow1) > len(flow2):
            print("Videos switched due to length (first has to be shorter than second).")
            self.flows[0] = flow2
            self.flows[1] = flow1

            self.videos[0] = video2
            self.videos[1] = video1

    def calc_correlation(self):
        print("Calculating correlation of flows.")

        flow1 = self.flows[0]
        flow2 = self.flows[1]

        len1 = len(flow1)
        len2 = len(flow2)

        if self.overlay > len1 or self.overlay > len2:
            print("Overlay too big, automatically decreased to default value 1000.")
            self.overlay = 1000

        correlation = np.empty((len1 + len2 - 2 * self.overlay + 1, 4))

        for i in range(self.overlay, len1 + len2 - self.overlay + 1):
            if i <= len1:
                correlation[i - self.overlay][0], _ = pearsonr(flow1[len1 - i:, 0], flow2[:i, 0])
                correlation[i - self.overlay][1], _ = pearsonr(flow1[len1 - i:, 1], flow2[:i, 1])
                correlation[i - self.overlay][2], _ = pearsonr(flow1[len1 - i:, 2], flow2[:i, 2])
                correlation[i - self.overlay][3], _ = pearsonr(flow1[len1 - i:, 3], flow2[:i, 3])
            elif i <= len2:
                correlation[i - self.overlay][0], _ = pearsonr(flow1[:, 0], flow2[i - len1:i, 0])
                correlation[i - self.overlay][1], _ = pearsonr(flow1[:, 1], flow2[i - len1:i, 1])
                correlation[i - self.overlay][2], _ = pearsonr(flow1[:, 2], flow2[i - len1:i, 2])
                correlation[i - self.overlay][3], _ = pearsonr(flow1[:, 3], flow2[i - len1:i, 3])
            else:
                correlation[i - self.overlay][0], _ = pearsonr(flow1[:len1 - (i - len2), 0], flow2[i - len1:, 0])
                correlation[i - self.overlay][1], _ = pearsonr(flow1[:len1 - (i - len2), 1], flow2[i - len1:, 1])
                correlation[i - self.overlay][2], _ = pearsonr(flow1[:len1 - (i - len2), 2], flow2[i - len1:, 2])
                correlation[i - self.overlay][3], _ = pearsonr(flow1[:len1 - (i - len2), 3], flow2[i - len1:, 3])

        return correlation

    def calc_difference(self, correlation):
        flow1 = self.flows[0]
        flow2 = self.flows[1]

        sum_correlation = np.abs(correlation).sum(axis=1)
        best_match = np.argpartition(sum_correlation, -1)[-1:][0]

        print(f"Length of first video: {len(flow1)}.")
        print(f"Length of second video: {len(flow2)}.")
        print(f"Overlay of videos taken into account: {self.overlay}.")
        print(f"Number of possible video combinations: {len(correlation)}.")
        print(f"Index of best match: {best_match}.")

        self.differences.append(self.overlay + best_match - len(flow1))

    def get_fps(self):
        print("Getting fps.")
        for video in self.videos:
            cap = cv2.VideoCapture(f'{self.directory}/{video}')
            self.fps.append(cap.get(cv2.CAP_PROP_FPS))
            print(f"Video {video} has {self.fps[-1]} fps.")

    def calc_cuts(self):
        new_len1 = len(self.flows[0])
        new_len2 = len(self.flows[1])

        self.get_fps()

        difference = self.differences[1]

        if difference < 0:
            new_len1 -= -difference
            print(f"Cut video {self.videos[0]} from {get_timestamp_from_seconds(-difference / self.fps[0])}")
        elif difference > 0:
            new_len2 -= difference
            print(f"Cut video {self.videos[1]} from {get_timestamp_from_seconds(difference / self.fps[1])}")
        else:
            print("Videos start at the same time.")

        if new_len1 > new_len2:
            print(f"Cut video {self.videos[0]} after (duration) {get_timestamp_from_seconds(new_len2 / self.fps[0])}")
        elif new_len1 < new_len2:
            print(f"Cut video {self.videos[1]} after (duration) {get_timestamp_from_seconds(new_len1 / self.fps[1])}")
        else:
            print("Videos end at the same time.")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        'directory',
        type=str,
        help="Path to the directory with videos (without slash at the end).",
    )
    parser.add_argument(
        '-o', '--overlay',
        type=int,
        default=1000,
        help="Number of frames that have to overlay."
    )
    parser.add_argument(
        '-l', '--load',
        action='store_true',
        help="Use when optical flow has already been calculated and can only be loaded."
    )
    args = parser.parse_args()

    video_synchronizer = VideoSynchronizer(args.directory, args.overlay)

    if args.load:
        video_synchronizer.load_flows()
    else:
        video_synchronizer.calculate_flows()

    video_synchronizer.check_length()

    correlation = video_synchronizer.calc_correlation()
    video_synchronizer.calc_difference(correlation)
    video_synchronizer.calc_cuts()


if __name__ == "__main__":
    main()
