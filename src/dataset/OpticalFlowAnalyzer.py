"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for analyzing optical flow data from videos.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 10. 07. 2022
"""

import numpy as np
from scipy.stats import normaltest
import matplotlib.pyplot as plt

from pathlib import Path
import cv2
import flow_vis

from src.dataset.utils.params import parse_arguments


def show_info(flow):
    print(f"- Flow Info")
    print(f"-- Minimum: {np.min(flow):,.01f}")
    print(f"-- 0.25: {np.quantile(flow, 0.25):,.01f}")
    print(f"-- Median: {np.median(flow):,.01f}")
    print(f"-- 0.75: {np.quantile(flow, 0.75):,.01f}")
    print(f"-- Maximum: {np.max(flow):,.01f}")
    print(f"-- Interquartile Range: {np.subtract(*np.percentile(flow, [75, 25])):,.01f}")


def show_deciles(flow):
    print(f"- Decile Analysis")
    print(f"-- 0.0: {np.percentile(flow, 0):,.01f}")
    print(f"-- 0.1: {np.percentile(flow, 10):,.01f}")
    print(f"-- 0.2: {np.percentile(flow, 20):,.01f}")
    print(f"-- 0.3: {np.percentile(flow, 30):,.01f}")
    print(f"-- 0.4: {np.percentile(flow, 40):,.01f}")
    print(f"-- 0.5: {np.percentile(flow, 50):,.01f}")
    print(f"-- 0.6: {np.percentile(flow, 60):,.01f}")
    print(f"-- 0.7: {np.percentile(flow, 70):,.01f}")
    print(f"-- 0.8: {np.percentile(flow, 80):,.01f}")
    print(f"-- 0.9: {np.percentile(flow, 90):,.01f}")
    print(f"-- 1.0: {np.percentile(flow, 100):,.01f}")


def test_normality(flow):
    print(f"- Normality Test")
    alpha = 1e-3
    k2, p = normaltest(flow)
    print(f"-- k2 = {k2:.03f}")
    print(f"-- p = {p:.07f} {'<' if p < alpha else '>'} alpha = {alpha} "
          f"=> Data {'not' if p < alpha else 'may be'} normally distributed.")
    print(f"-- Mean: {np.mean(flow):.01f}")
    print(f"-- Variance: {np.std(flow):.01f}")


def plot_flow(flow):
    plt.hist(flow, bins=100, range=[np.min(flow), np.max(flow)], log=True)
    plt.show()


def detect_movement(flow):
    print(f"- Movement detection")
    median = np.median(flow)
    print(f"-- Frames greater than Median: {(flow > median).sum()}")
    d6 = np.percentile(flow, 60)
    print(f"-- Frames greater than D6: {(flow > d6).sum()}")
    d7 = np.percentile(flow, 70)
    print(f"-- Frames greater than D7: {(flow > d7).sum()}")
    q3 = np.quantile(flow, 0.75)
    print(f"-- Frames greater than Q3: {(flow > q3).sum()}")
    d8 = np.percentile(flow, 80)
    print(f"-- Frames greater than D8: {(flow > d8).sum()}")
    d9 = np.percentile(flow, 90)
    print(f"-- Frames greater than D9: {(flow > d9).sum()}")
    p95 = np.percentile(flow, 95)
    print(f"-- Frames greater than P95: {(flow > p95).sum()}")


class OpticalFlowAnalyzer:
    def __init__(self, file):
        flow = np.load(file)
        self.up_flow = flow[:, 0]
        self.down_flow = flow[:, 1]
        self.left_flow = flow[:, 2]
        self.right_flow = flow[:, 3]

    def analyze(self):
        print(f"Analyzing optical flow...")
        print(f"- Number of frames: {len(self.up_flow)}")

        print(f"\nUpward motion")
        plot_flow(self.up_flow)
        show_info(self.up_flow)
        show_deciles(self.up_flow)
        test_normality(self.up_flow)
        detect_movement(self.up_flow)

        print(f"\nDownward motion")
        plot_flow(self.down_flow)
        show_info(self.down_flow)
        show_deciles(self.down_flow)
        test_normality(self.down_flow)
        detect_movement(self.down_flow)

        print(f"\nLeftward motion")
        plot_flow(self.left_flow)
        show_info(self.left_flow)
        show_deciles(self.left_flow)
        test_normality(self.left_flow)
        detect_movement(self.left_flow)

        print(f"\nRightward motion")
        plot_flow(self.right_flow)
        show_info(self.right_flow)
        show_deciles(self.right_flow)
        test_normality(self.right_flow)
        detect_movement(self.right_flow)


def visualize(file):
    cap = cv2.VideoCapture(file)
    file_path = Path(file)

    counter = 0
    ret, prev_frame = cap.read()
    while ret:
        ret, current_frame = cap.read()

        if counter in [312, 386, 1127, 1664]:
            flow = cv2.optflow.calcOpticalFlowDenseRLOF(prev_frame, current_frame, None)
            flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=True)
            cv2.imwrite(f"{file_path.parent / f'flow{counter:09d}.png'}", flow_color)
            cv2.imwrite(f"{file_path.parent / f'frame{counter:09d}.png'}", current_frame)

        counter += 1
        prev_frame = current_frame


def main():
    args = parse_arguments()

    # batch_provider = OpticalFlowAnalyzer(args.location)
    # batch_provider.analyze()

    visualize(args.location)


if __name__ == "__main__":
    main()
