"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for synchronization of multiple videos of the same scene.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 24. 02. 2022
Source: ffmpeg commands
    (https://stackoverflow.com/questions/11552565/vertically-or-horizontally-stack-mosaic-several-videos-using-ffmpeg)
"""

from pathlib import Path
import re

import numpy as np
from scipy.stats import pearsonr

import OpticalFlowCalculator
from src.utils.params import parse_arguments
import src.utils.video_edit as video_edit


class VideoSynchronizer:
    def __init__(self, directory, overlay=1000):
        self.overlay = overlay
        self.videos = list(Path(directory).glob('*.mp4'))
        self.flows = []
        self.flow1 = 0
        self.flow2 = 1
        self.differences = [0]
        self.shortest = 0

    def load_flows(self):
        print("Loading flows...")

        for video in self.videos:
            self.flows.append(np.load(video.with_suffix(".npy")))
            print(f"- Flow of video {video} loaded.")

    def calculate_flows(self):
        print("Calculating flows...")

        optical_flow_calc = OpticalFlowCalculator.OpticalFlowCalculator(self.videos)

        for video in self.videos:
            flow = optical_flow_calc.process_video(video)
            self.flows.append(flow)
            print(f"- Flow from video {video} calculated.")

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
        std_thresh = 10
        number_of_compared = 10

        sum_correlation = np.abs(correlation).sum(axis=1)
        matches = sum_correlation.argsort()[-number_of_compared:]
        print(f"Found matches on indices: {matches[::-1]}, with standard deviation: {np.std(matches)}.")

        while np.std(matches) > std_thresh:
            print("Synchronization might fail due to not precise correlation - cleaning of results.")
            matches = matches[abs(matches - matches.mean()) <= matches.std()]
            print(f"Cleaned matches are on indices: {matches[::-1]}, with standard deviation: {np.std(matches)}.")

        best_match = matches[-1]
        print(f"Best match: {best_match}")

        if self.flow1 > self.flow2:  # Flows were switched.
            self.differences.append(-(self.overlay + best_match - len(self.flows[self.flow1])))
        else:
            self.differences.append(self.overlay + best_match - len(self.flows[self.flow1]))

    def calc_cuts(self):
        latest = min(self.differences)
        new_lengths = []
        for i in range(len(self.differences)):
            self.differences[i] -= latest
            new_lengths.append(len(self.flows[i]) - self.differences[i])

        self.shortest = min(new_lengths)

    def synchronize_videos(self):
        for flow_index in range(1, len(self.flows)):
            self.flow1 = 0
            self.flow2 = flow_index

            print(f"\nUsing flows {self.flow1} and {self.flow2}.")

            # Check if flow1 is shorter than flow2, if not, switch indices.
            self.check_length()

            correlation = self.calc_correlation()
            self.calc_difference(correlation)

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

        print("\nPreparing commands to cut the videos...")
        print("Video cutting suggestions:")
        commands = ""

        for i in range(len(self.differences)):
            start_secs = video_edit.get_seconds_from_frame(self.videos[i], self.differences[i])
            duration_secs = video_edit.get_seconds_from_frame(self.videos[i], self.shortest)
            start_time = video_edit.get_timestamp_from_seconds(start_secs)
            duration = video_edit.get_timestamp_from_seconds(duration_secs)

            print(f"- Cut video {self.videos[i]} from {start_time} for (duration) {duration}.")

            commands += f'\nffmpeg \\\n'
            commands += f'-ss {start_time} \\\n'
            commands += f'-i {self.videos[i]} \\\n'
            commands += f'-t {duration} \\\n'
            commands += f'-codec:v libx264 \\\n'
            commands += f'{self.videos[i].parent / (self.videos[i].stem + "_synced" + self.videos[i].suffix)}\n'

        if 2 <= len(self.differences) <= 4:
            commands += f'\nffmpeg \\\n'
            for i in range(len(self.differences)):
                commands += f'-i {self.videos[i].parent / (self.videos[i].stem + "_synced" + self.videos[i].suffix)} '
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
            output_name = re.sub(r'cam\d_normalized', 'stacked', f'{self.videos[0]}')
            commands += f'-codec:v libx264 \\\n'
            commands += f'{output_name}\n'

        return commands

    def export_flows(self):
        print("Exporting synced flows...")
        for i in range(len(self.flows)):
            print(f'- Flow {self.videos[i].stem + "_synced.npy"} exported.')
            np.save(f'{self.videos[i].parent / (self.videos[i].stem + "_synced.npy")}',
                    self.flows[i][self.differences[i]:self.differences[i] + self.shortest])


def main():
    args = parse_arguments()

    video_synchronizer = VideoSynchronizer(args.location, args.overlay)

    if args.load:
        video_synchronizer.load_flows()
    else:
        video_synchronizer.calculate_flows()

    video_synchronizer.synchronize_videos()
    video_synchronizer.calc_cuts()
    video_synchronizer.export_flows()

    command = video_synchronizer.get_cut_commands()
    if args.script:
        video_edit.create_script(command)
        print("- Synchronization script created.")
    else:
        video_edit.execute_command(command)
        print("- Videos synchronized.")


if __name__ == "__main__":
    main()
