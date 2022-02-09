"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for synchronization of multiple videos of the same scene.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 9. 2. 2022
Source: ffmpeg commands
    (https://stackoverflow.com/questions/11552565/vertically-or-horizontally-stack-mosaic-several-videos-using-ffmpeg)
"""

import os
import re

import numpy as np
from scipy.stats import pearsonr
import cv2

import OpticalFlowCalculator
from src.utils.params import parse_arguments


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
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{milliseconds:03d}"


def cut_videos(commands):
    os.system(commands)
    print("- Videos cut.")


def create_script(commands):
    with open(f'cut_videos.sh', 'w') as script:
        script.write("# Script automatically created with VideoSynchronizer module.\n")
        script.write("# Author: Daniel Konecny (xkonec75).\n")
        script.write(commands)

        print("- Synchronization script created.")


class VideoSynchronizer:
    def __init__(self, directory, overlay=1000):
        self.directory = directory
        self.overlay = overlay
        self.files = []
        self.flows = []
        self.fps = []
        self.flow1 = 0
        self.flow2 = 1
        self.differences = [0]
        self.shortest = 0

    def load_flows(self):
        print("Loading flows...")

        for file in os.listdir(self.directory):
            if file.endswith('.npy'):
                self.flows.append(np.load(f'{self.directory}/{file}'))
                self.files.append(file.replace('.npy', ''))
                print(f"- Flow {file} loaded.")

    def calculate_flows(self):
        print("Calculating flows...")

        optical_flow_calc = OpticalFlowCalculator.OpticalFlowCalculator(self.directory)

        for file in os.listdir(self.directory):
            if file.endswith('.mp4'):
                flow = optical_flow_calc.process_video(file)
                self.flows.append(flow)
                self.files.append(file.replace('.mp4', ''))
                print(f"- Flow from video {file} calculated.")

    def check_length(self):
        print("Checking length of flows...")

        if len(self.flows[self.flow1]) > len(self.flows[self.flow2]):
            print("- Videos switched due to length (first has to be shorter than second).")
            temp = self.flow1
            self.flow1 = self.flow2
            self.flow2 = temp

    def calc_correlation(self):
        print("Calculating correlation of flows...")

        len1 = len(self.flows[self.flow1])
        len2 = len(self.flows[self.flow2])

        if self.overlay > len1 or self.overlay > len2:
            print("- Overlay too big, automatically decreased to default value 1000.")
            self.overlay = 1000

        correlation = np.empty((len1 + len2 - 2 * self.overlay + 1, 4))

        for i in range(self.overlay, len1 + len2 - self.overlay + 1):
            if i <= len1:
                correlation[i - self.overlay][0], _ = pearsonr(self.flows[self.flow1][len1 - i:, 0],
                                                               self.flows[self.flow2][:i, 0])
                correlation[i - self.overlay][1], _ = pearsonr(self.flows[self.flow1][len1 - i:, 1],
                                                               self.flows[self.flow2][:i, 1])
                correlation[i - self.overlay][2], _ = pearsonr(self.flows[self.flow1][len1 - i:, 2],
                                                               self.flows[self.flow2][:i, 2])
                correlation[i - self.overlay][3], _ = pearsonr(self.flows[self.flow1][len1 - i:, 3],
                                                               self.flows[self.flow2][:i, 3])
            elif i <= len2:
                correlation[i - self.overlay][0], _ = pearsonr(self.flows[self.flow1][:, 0],
                                                               self.flows[self.flow2][i - len1:i, 0])
                correlation[i - self.overlay][1], _ = pearsonr(self.flows[self.flow1][:, 1],
                                                               self.flows[self.flow2][i - len1:i, 1])
                correlation[i - self.overlay][2], _ = pearsonr(self.flows[self.flow1][:, 2],
                                                               self.flows[self.flow2][i - len1:i, 2])
                correlation[i - self.overlay][3], _ = pearsonr(self.flows[self.flow1][:, 3],
                                                               self.flows[self.flow2][i - len1:i, 3])
            else:
                correlation[i - self.overlay][0], _ = pearsonr(self.flows[self.flow1][:len1 - (i - len2), 0],
                                                               self.flows[self.flow2][i - len1:, 0])
                correlation[i - self.overlay][1], _ = pearsonr(self.flows[self.flow1][:len1 - (i - len2), 1],
                                                               self.flows[self.flow2][i - len1:, 1])
                correlation[i - self.overlay][2], _ = pearsonr(self.flows[self.flow1][:len1 - (i - len2), 2],
                                                               self.flows[self.flow2][i - len1:, 2])
                correlation[i - self.overlay][3], _ = pearsonr(self.flows[self.flow1][:len1 - (i - len2), 3],
                                                               self.flows[self.flow2][i - len1:, 3])

        return correlation

    def calc_difference(self, correlation):
        sum_correlation = np.abs(correlation).sum(axis=1)

        best_match = np.argpartition(sum_correlation, -1)[-1:][0]

        if self.flow1 > self.flow2:  # Flows were switched.
            self.differences.append(-(self.overlay + best_match - len(self.flows[self.flow1])))
        else:
            self.differences.append(self.overlay + best_match - len(self.flows[self.flow1]))

    def get_fps(self):
        print("Getting fps...")
        for video in self.files:
            cap = cv2.VideoCapture(f'{self.directory}/{video}.mp4')
            self.fps.append(cap.get(cv2.CAP_PROP_FPS))
            print(f"- Video {video}.mp4 has {self.fps[-1]} fps.")

    def calc_cuts(self):
        print("\nVideo cutting suggestions:")
        self.get_fps()

        latest = min(self.differences)
        new_lengths = []
        for i in range(len(self.differences)):
            self.differences[i] -= latest
            new_lengths.append(len(self.flows[i]) - self.differences[i])

        self.shortest = min(new_lengths)

        for i in range(len(self.differences)):
            start_time = get_timestamp_from_seconds(self.differences[i] / self.fps[i])
            duration = get_timestamp_from_seconds(self.shortest / self.fps[i])
            print(f"- Cut video {self.directory}/{self.files[i]}.mp4 from {start_time} for (duration) {duration}.")

    def synchronize_videos(self):
        for flow_index in range(1, len(self.flows)):
            self.flow1 = 0
            self.flow2 = flow_index

            print(f"\nUsing flows {self.flow1} and {self.flow2}.")

            # Check if flow1 is shorter than flow2, if not, switch indices.
            self.check_length()

            correlation = self.calc_correlation()
            self.calc_difference(correlation)

        self.calc_cuts()

    def get_cut_commands(self):
        """
        ffmpeg command to cut videos:
            ffmpeg
            -ss hh:mm:ss.mmm (start time)
            -i input_file
            -t hh:mm:ss.mmm (duration)
            -codec:v libx264
            output_file

        ffmpeg command to stack videos:
            ffmpeg
            -i input_file
            -filter_complex hstack (2 videos)
            -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" -map "[v]" (3 videos)
            -filter_complex "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" -map "[v]" (4 videos)
            -codec:v libx264
            output_file

        :return: String with constructed ffmpeg command to synchronize and stack the videos.
        """

        print("Preparing commands to cut the videos...")
        commands = ""

        for i in range(len(self.differences)):
            start_time = get_timestamp_from_seconds(self.differences[i] / self.fps[i])
            duration = get_timestamp_from_seconds(self.shortest / self.fps[i])

            commands += f'\nffmpeg \\\n'
            commands += f'-ss {start_time} \\\n'
            commands += f'-i {self.directory}/{self.files[i]}.mp4 \\\n'
            commands += f'-t {duration} \\\n'
            commands += f'-codec:v libx264 \\\n'
            commands += f'{self.directory}/{self.files[i]}_synced.mp4\n'

        if 2 <= len(self.differences) <= 4:
            commands += f'\nffmpeg \\\n'
            for i in range(len(self.differences)):
                commands += f'-i {self.directory}/{self.files[i]}_synced.mp4 '
            commands += f'\\\n'

        if len(self.differences) == 2:
            commands += f'-filter_complex hstack \\\n'
        elif len(self.differences) == 3:
            commands += f'-filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" \\\n'
            commands += f'-map "[v]" \\\n'
        elif len(self.differences) == 4:
            commands += f'-filter_complex "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" \\\n'
            commands += f'-map "[v]" \\\n'

        if 2 <= len(self.differences) <= 4:
            output_name = re.sub(r"video\d_normalized", "stacked", f"{self.files[0]}.mp4")
            commands += f'-codec:v libx264 \\\n'
            commands += f'{self.directory}/{output_name}\n'

        return commands

    def export_flows(self):
        print("Exporting synced flows...")
        for index in range(len(self.flows)):
            print(f"- Flow {self.files[index]}_synced.npy exported.")
            np.save(f'{self.directory}/{self.files[index]}_synced.npy',
                    self.flows[index][self.differences[index]:self.differences[index] + self.shortest])


def main():
    args = parse_arguments()

    video_synchronizer = VideoSynchronizer(args.location, args.overlay)

    if args.load:
        video_synchronizer.load_flows()
    else:
        video_synchronizer.calculate_flows()

    video_synchronizer.synchronize_videos()
    video_synchronizer.export_flows()

    commands = video_synchronizer.get_cut_commands()
    if args.script:
        create_script(commands)
    else:
        cut_videos(commands)


if __name__ == "__main__":
    main()
