"""Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for analyzing optical flow data from videos.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 15. 10. 2021
"""


from argparse import ArgumentParser

import numpy as np
from scipy.stats import normaltest
from matplotlib import pyplot


def show_info(flow):
    print(f"- Flow Info")
    print(f"-- Minimum: {np.min(flow):.01f}")
    print(f"-- 0.25: {np.quantile(flow, 0.25):.01f}")
    print(f"-- Median: {np.median(flow):.01f}")
    print(f"-- 0.75: {np.quantile(flow, 0.75):.01f}")
    print(f"-- Maximum: {np.max(flow):.01f}")
    print(f"-- Interquartile Range: {np.subtract(*np.percentile(flow, [75, 25])):.01f}")


def show_deciles(flow):
    print(f"- Decile Analysis")
    print(f"-- 0.0: {np.percentile(flow, 0):.01f}")
    print(f"-- 0.1: {np.percentile(flow, 10):.01f}")
    print(f"-- 0.2: {np.percentile(flow, 20):.01f}")
    print(f"-- 0.3: {np.percentile(flow, 30):.01f}")
    print(f"-- 0.4: {np.percentile(flow, 40):.01f}")
    print(f"-- 0.5: {np.percentile(flow, 50):.01f}")
    print(f"-- 0.6: {np.percentile(flow, 60):.01f}")
    print(f"-- 0.7: {np.percentile(flow, 70):.01f}")
    print(f"-- 0.8: {np.percentile(flow, 80):.01f}")
    print(f"-- 0.9: {np.percentile(flow, 90):.01f}")
    print(f"-- 1.0: {np.percentile(flow, 100):.01f}")


def test_normality(flow):
    print(f"- Normality Test")
    alpha = 1e-3
    k2, p = normaltest(flow)
    print(f"-- k2={k2:.03f}")
    print(f"-- p={p:.03f}")
    if p < alpha:
        print(f"--> Data not normally distributed.")
    else:
        print(f"--> Data may be normally distributed.")
    print(f"-- Mean: {np.mean(flow):.01f}")
    print(f"-- Variance: {np.std(flow):.01f}")


def plot_flow(flow):
    pyplot.hist(flow, bins=100, range=[np.min(flow), np.max(flow)], log=True)
    pyplot.show()


def detect_movement(flow):
    print(f"- Movement detection")
    d6 = np.percentile(flow, 60)
    print(f"-- Frames greater than D6: {(flow > d6).sum()}")
    d7 = np.percentile(flow, 70)
    print(f"-- Frames greater than D7: {(flow > d7).sum()}")
    q3 = np.quantile(flow, 0.75)
    print(f"-- Frames greater than Q3: {(flow > q3).sum()}")
    d8 = np.percentile(flow, 80)
    print(f"-- Frames greater than D8: {(flow > d8).sum()}")
    mean = np.mean(flow)
    print(f"-- Frames greater than Mean: {(flow > mean).sum()}")


class OpticalFlowAnalyzer:
    def __init__(self, file):
        flow = np.load(file)
        self.up_flow = flow[:, 0]
        self.down_flow = flow[:, 1]
        self.right_flow = flow[:, 2]
        self.left_flow = flow[:, 3]

    def analyze(self):
        print(f"Analyzing optical flow...")
        print(f"- Number of frames: {len(self.up_flow)}")

        print(f"\nUpward motion")
        # plot_flow(self.up_flow)
        show_info(self.up_flow)
        show_deciles(self.up_flow)
        # test_normality(self.up_flow)
        detect_movement(self.up_flow)

        # print(f"\nDownward motion")
        # print_flow_info(self.down_flow)
        # print(f"\nLeftward motion")
        # print_flow_info(self.left_flow)
        # print(f"\nRightward motion")
        # print_flow_info(self.right_flow)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        'file',
        type=str,
        help="Path to optical flow saved as NumPy nd-array in a file.",
    )
    args = parser.parse_args()

    batch_provider = OpticalFlowAnalyzer(args.file)
    batch_provider.analyze()


if __name__ == "__main__":
    main()
