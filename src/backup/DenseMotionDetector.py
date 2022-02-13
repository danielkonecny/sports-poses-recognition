"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Detects motion in video with dense optical flow.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 13. 02. 2022
"""

import numpy as np

COMMON_INFO_IDX = 0


def get_movement_index(flows, move_thresh, steps, frame_skip):
    up_flow_thresh = down_flow_thresh = left_flow_thresh = right_flow_thresh = np.empty((len(flows)))

    for flow_idx in range(len(flows)):
        up_flow_thresh[flow_idx] = np.percentile(flows[flow_idx][:, 0], move_thresh)
        down_flow_thresh[flow_idx] = np.percentile(flows[flow_idx][:, 1], move_thresh)
        left_flow_thresh[flow_idx] = np.percentile(flows[flow_idx][:, 2], move_thresh)
        right_flow_thresh[flow_idx] = np.percentile(flows[flow_idx][:, 3], move_thresh)

    for index in range(len(flows[COMMON_INFO_IDX])):
        if index + steps * frame_skip > len(flows[COMMON_INFO_IDX]):
            break

        # Requires movement above threshold between all steps for all cameras.
        enough_movement = True
        for flow_idx in range(len(flows)):
            for step_idx in range(steps - 1):  # Movement in the last step is not needed.
                if not (flows[flow_idx][index + step_idx * frame_skip][0] > up_flow_thresh[flow_idx] or
                        flows[flow_idx][index + step_idx * frame_skip][1] > down_flow_thresh[flow_idx] or
                        flows[flow_idx][index + step_idx * frame_skip][2] > left_flow_thresh[flow_idx] or
                        flows[flow_idx][index + step_idx * frame_skip][3] > right_flow_thresh[flow_idx]):
                    enough_movement = False

        if enough_movement:
            yield index
