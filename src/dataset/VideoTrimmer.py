"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Trims video at selected start and end points.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 13. 02. 2022
Source: use trackbar to select star and end points
    (https://stackoverflow.com/questions/21983062/in-python-opencv-is-there-a-way-to-quickly-scroll-through-frames-of-a-video-all)
"""

from pathlib import Path
from functools import partial

import cv2

from src.utils.params import parse_arguments
import src.utils.video_edit as video_edit


def on_change(video, video_name, trackbar_value):
    video.set(cv2.CAP_PROP_POS_FRAMES, trackbar_value)
    err, img = video.read()
    cv2.imshow(video_name, img)
    pass


def select_points(video, video_name):
    print("Select start and end point.\n"
          "Press Enter to confirm.\n"
          "Press Space to run the video.\n")

    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.namedWindow(video_name)

    # partial is used to pass arguments to a callback function.
    cv2.createTrackbar('Start', video_name, 0, length, partial(on_change, video, video_name))
    cv2.createTrackbar('End', video_name, 100, length, partial(on_change, video, video_name))

    on_change(video, video_name, 0)
    cv2.waitKey()

    start = cv2.getTrackbarPos('Start', video_name)
    end = cv2.getTrackbarPos('End', video_name)
    if start >= end:
        raise Exception("Start must be less than end.")

    video.set(cv2.CAP_PROP_POS_FRAMES, start)
    while video.isOpened():
        err, img = video.read()
        if video.get(cv2.CAP_PROP_POS_FRAMES) >= end:
            break
        cv2.imshow(video_name, img)
        k = cv2.waitKey(10) & 0xff
        if k == 13:
            break

    return start, end


def get_trim_command(video_path, start, duration):
    """
    ffmpeg command to cut videos:
        ffmpeg
        -ss hh:mm:ss.mmm (start time)
        -i input_file
        -t hh:mm:ss.mmm (duration)
        -codec:v libx264
        output_file

    :return: String with constructed ffmpeg command to trim the video.
    """

    print("\nPreparing command to trim the video...")

    command = f'\nffmpeg \\\n'
    command += f'-ss {start} \\\n'
    command += f'-i {video_path} \\\n'
    command += f'-t {duration} \\\n'
    command += f'-codec:v libx264 \\\n'
    command += f'{video_path.parent / (video_path.stem + "_trimmed" + video_path.suffix)}\n'

    return command


def main():
    args = parse_arguments()

    video_path = Path(args.location)
    video_name = str(video_path.name)

    video = cv2.VideoCapture(str(video_path.resolve()))

    start, end = select_points(video, video_name)
    start_secs = video_edit.get_seconds_from_frame(video, start)
    end_secs = video_edit.get_seconds_from_frame(video, end)
    start_time = video_edit.get_timestamp_from_seconds(start_secs)
    duration = video_edit.get_timestamp_from_seconds(end_secs - start_secs)

    command = get_trim_command(video_path, start_time, duration)

    if args.script:
        video_edit.create_script(command)
        print("- Trim script created.")
    else:
        video_edit.execute_command(command)
        print("- Video trimmed.")


if __name__ == "__main__":
    main()
