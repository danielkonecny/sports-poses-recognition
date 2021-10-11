import numpy as np
from scipy.stats import pearsonr


def main():
    fps = 29.97
    video1 = 'out/flow_20211009_0.npy'
    video2 = 'out/flow_20211009_2.npy'

    flow1 = np.load(video1)
    flow2 = np.load(video2)

    # flow1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # flow2 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    if len(flow1) > len(flow2):
        print("Videos switched due to length (first has to be shorter than second).")
        temp = flow1
        flow1 = flow2
        flow2 = temp

        temp = video1
        video1 = video2
        video2 = temp

    len1 = len(flow1)
    len2 = len(flow2)

    # Number of frames that surely overlay.
    overlay = 5000

    correlation = np.empty((len1 + len2 - 2*overlay + 1, 4))

    for i in range(overlay, len1 + len2 - overlay + 1):
        if i <= len1:
            correlation[i - overlay][0], _ = pearsonr(flow1[len1 - i:, 0], flow2[:i, 0])
            correlation[i - overlay][1], _ = pearsonr(flow1[len1 - i:, 1], flow2[:i, 1])
            correlation[i - overlay][2], _ = pearsonr(flow1[len1 - i:, 2], flow2[:i, 2])
            correlation[i - overlay][3], _ = pearsonr(flow1[len1 - i:, 3], flow2[:i, 3])
            # print(f"1 size: {flow1[len1 - i:].shape} - {flow2[:i].shape}")
            # print(f"1 elements: {flow1[len1 - i:]} - {flow2[:i]}")
        elif i <= len2:
            correlation[i - overlay][0], _ = pearsonr(flow1[:, 0], flow2[i - len1:i, 0])
            correlation[i - overlay][1], _ = pearsonr(flow1[:, 1], flow2[i - len1:i, 1])
            correlation[i - overlay][2], _ = pearsonr(flow1[:, 2], flow2[i - len1:i, 2])
            correlation[i - overlay][3], _ = pearsonr(flow1[:, 3], flow2[i - len1:i, 3])
            # print(f"2 size: {flow1[:].shape} - {flow2[i - len1:i].shape}")
            # print(f"2 elements: {flow1[:]} - {flow2[i - len1:i]}")
        else:
            correlation[i - overlay][0], _ = pearsonr(flow1[:len1 - (i - len2), 0], flow2[i - len1:, 0])
            correlation[i - overlay][1], _ = pearsonr(flow1[:len1 - (i - len2), 1], flow2[i - len1:, 1])
            correlation[i - overlay][2], _ = pearsonr(flow1[:len1 - (i - len2), 2], flow2[i - len1:, 2])
            correlation[i - overlay][3], _ = pearsonr(flow1[:len1 - (i - len2), 3], flow2[i - len1:, 3])
            # print(f"3 size: {flow1[:len1 - (i - len2)].shape} - {flow2[i - len1:].shape}")
            # print(f"3 elements: {flow1[:len1 - (i - len2)]} - {flow2[i - len1:]}")

    sum_correlation = np.abs(correlation).sum(axis=1)
    best_match = np.argpartition(sum_correlation, -1)[-1:][0]

    print(f"Length of first video: {len1}.")
    print(f"Length of second video: {len2}.")
    print(f"Overlay of videos taken into account: {overlay}.")
    print(f"Number of possible video combinations: {len(correlation)}.")
    print(f"Index of best match: {best_match}.")

    new_len1 = len1
    new_len2 = len2

    difference = overlay + best_match - len1
    if difference < 0:
        difference = -difference
        new_len1 -= difference
        print(f"Cut video {video1} from ", end="")
    elif difference > 0:
        new_len2 -= difference
        print(f"Cut video {video2} from ", end="")
    else:
        print("Videos start at the same time.")

    seconds = difference / fps
    milliseconds = int((seconds % 1)*1000)
    hours = 0
    minutes = 0
    if seconds > 3600:
        hours = int(seconds // 3600)
        seconds %= 3600
    if seconds > 60:
        minutes = int(seconds // 60)
        seconds %= 60
    print(f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{milliseconds:03d}.")

    cut_after = 0
    if new_len1 > new_len2:
        cut_after = new_len2
        print(f"Cut video {video1} after (duration) ", end="")
    elif new_len1 < new_len2:
        cut_after = new_len1
        print(f"Cut video {video2} after (duration) ", end="")
    else:
        print("Videos end at the same time.")

    seconds = cut_after / fps
    milliseconds = int((seconds % 1) * 1000)
    hours = 0
    minutes = 0
    if seconds > 3600:
        hours = int(seconds // 3600)
        seconds %= 3600
    if seconds > 60:
        minutes = int(seconds // 60)
        seconds %= 60
    print(f"{hours:02d}:"
          f"{minutes:02d}:"
          f"{int(seconds):02d}."
          f"{milliseconds:03d}.")


if __name__ == "__main__":
    main()
