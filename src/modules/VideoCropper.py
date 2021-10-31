"""Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for cropping of video to a selected area.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 31. 10. 2021
Source: https://stackoverflow.com/questions/6136588/image-cropping-using-python/8696558
"""

import os
from argparse import ArgumentParser
import numpy as np
import cv2

import contextlib

with contextlib.redirect_stdout(None):
    import pygame


# TODO - Adjust to be able to have any result ratio, not just square.
# TODO - Form as an interface to get the right dimensions


def crop_video(crop_command):
    os.system(crop_command)
    print("- Video cropped.")


def create_script(crop_command):
    with open(f'crop_video.sh', 'w') as script:
        script.write("# Script automatically created with VideoCropper module.\n")
        script.write("# Author: Daniel Konecny (xkonec75).\n")
        script.write(crop_command)

        print("- Crop script created.")


class VideoCropper:
    def __init__(self, video, result_w, result_h):
        self.video = video
        self.result_w = result_w
        self.result_h = result_h
        self.px = None
        self.screen = None
        self.video_w = self.video_h = 0
        self.left = self.right = self.upper = self.lower = 0

    def display_image(self, top_left, prior):
        # Ensure that the rect always has positive width, height.
        x, y = top_left
        width = pygame.mouse.get_pos()[0] - top_left[0]
        height = pygame.mouse.get_pos()[1] - top_left[1]
        if width < 0:
            x += width
            width = abs(width)
        if height < 0:
            y += height
            height = abs(height)

        # Eliminate redundant drawing cycles (when mouse isn't moving).
        current = x, y, width, height
        if not (width and height):
            return current
        if current == prior:
            return current

        # Draw transparent box and blit it onto canvas.
        self.screen.blit(self.px, self.px.get_rect())
        display_im = pygame.Surface((width, height))
        display_im.fill((128, 128, 128))
        pygame.draw.rect(display_im, (32, 32, 32), display_im.get_rect(), 1)
        display_im.set_alpha(128)
        self.screen.blit(display_im, (x, y))
        pygame.display.flip()

        return x, y, width, height

    def set_crops(self):
        pygame.display.flip()

        top_left = bottom_right = prior = None
        n = 0
        while n != 1:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:
                    if not top_left:
                        top_left = event.pos
                    else:
                        bottom_right = event.pos
                        n = 1
            if top_left:
                prior = self.display_image(top_left, prior)

        crops = top_left + bottom_right
        self.left = crops[0]
        self.upper = crops[1]
        self.right = crops[2]
        self.lower = crops[3]

    def check_crop_direction(self):
        print("- Checking the direction of crop selection.")
        if self.right < self.left:
            print("-- Right and left coordinates swapped.")
            self.left, self.right = self.right, self.left
        if self.lower < self.upper:
            print("-- Upper and lower coordinates swapped.")
            self.lower, self.upper = self.upper, self.lower

    def check_crop_proportions(self):
        print("- Checking if crop selection is a square.")
        crop_width = self.right - self.left
        crop_height = self.lower - self.upper
        if crop_width > crop_height:
            print("-- Increasing height to form a square.")
            difference_cw_ch = crop_width - crop_height
            half_difference_cw_ch = difference_cw_ch // 2
            if difference_cw_ch % 2 == 0:
                self.upper -= half_difference_cw_ch
                self.lower += half_difference_cw_ch
            else:
                self.upper -= half_difference_cw_ch
                self.lower += half_difference_cw_ch + 1
        elif crop_width < crop_height:
            print("-- Increasing width to form a square.")
            difference_ch_cw = crop_height - crop_width
            half_difference_ch_cw = difference_ch_cw // 2
            if difference_ch_cw % 2 == 0:
                self.left -= half_difference_ch_cw
                self.right += half_difference_ch_cw
            else:
                self.left -= half_difference_ch_cw
                self.right += half_difference_ch_cw + 1

    def check_crop_size_to_result(self):
        print("- Checking if crop selection is large enough.")
        crop_size = self.right - self.left  # Width and height are now equal.
        if crop_size < self.result_w:  # Goal width and height are equal.
            print("-- Increasing crop selection size.")
            difference = self.result_w - crop_size
            half_difference = difference // 2
            if difference % 2 == 0:
                self.left -= half_difference
                self.right += half_difference
                self.upper -= half_difference
                self.lower += half_difference
            else:
                self.left -= half_difference
                self.right += half_difference + 1
                self.upper -= half_difference
                self.lower += half_difference + 1

    def check_crop_size_to_frame(self):
        print("- Checking if crop selection did not exceed the frame size.")
        crop_size = self.right - self.left  # Width and height are now equal.
        if crop_size > self.video_w or crop_size > self.video_h:
            difference = 0
            if crop_size > self.video_w:
                print("-- Width too high, decreasing both dimensions.")
                difference = crop_size - self.video_w
            if crop_size > self.video_h:
                print("-- Height too high, decreasing both dimensions.")
                difference = crop_size - self.video_h
            half_difference = difference // 2
            if difference % 2 == 0:
                self.left += half_difference
                self.right -= half_difference
                self.upper += half_difference
                self.lower -= half_difference
            else:
                self.left += half_difference
                self.right -= half_difference + 1
                self.upper += half_difference
                self.lower -= half_difference + 1

    def check_crop_position(self):
        print("- Checking if crop selection is not out of the frame.")
        if self.left < 0:
            print("-- Moving crop to the right.")
            self.right += -self.left
            self.left = 0
        elif self.right > self.video_w:
            print("-- Moving crop to the left.")
            self.left -= self.right - self.video_w
            self.right = self.video_w
        if self.upper < 0:
            print("-- Moving crop down.")
            self.lower += -self.upper
            self.upper = 0
        elif self.lower > self.video_h:
            print("-- Moving crop up.")
            self.upper -= self.lower - self.video_h
            self.lower = self.video_h

    def recalculate_crops(self):
        print("\nRecalculating crop area...")
        print("- Original coordinates and information")
        print(f"-- Frame width - {self.video_w}")
        print(f"-- Frame height - {self.video_h}")
        print(f"-- Left - {self.left}")
        print(f"-- Right - {self.right}")
        print(f"-- Upper - {self.upper}")
        print(f"-- Lower - {self.lower}")

        self.check_crop_direction()
        self.check_crop_proportions()
        self.check_crop_size_to_result()
        self.check_crop_size_to_frame()
        self.check_crop_position()

        print("- Recalculated coordinates")
        print(f"-- Left - {self.left}")
        print(f"-- Right - {self.right}")
        print(f"-- Upper - {self.upper}")
        print(f"-- Lower - {self.lower}")

    def get_image(self):
        print("Merging video to single image...")
        image_count = 5
        image_channels = 3

        video = cv2.VideoCapture(f'{self.video}')
        self.video_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        merged_image = np.zeros((self.video_h, self.video_w, image_channels))

        for image_index in range(1, image_count + 1):
            frame_to_use = image_index * frame_count // (image_count + 1)
            print(f"- Using frame: {frame_to_use} of {frame_count} frames.")
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_to_use)
            ret, frame = video.read()
            if not ret:
                print(f"-- Frame from video could not be obtained.")
            frame = frame.astype(np.float32) / 255.
            merged_image = np.add(merged_image, frame / image_count)

        merged_image = (merged_image * 255).astype(np.uint8)

        # pygame uses (w, h) and RGB, but OpenCV uses (h, w) and BGR image format.
        formatted_frame = cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        self.px = pygame.surfarray.make_surface(formatted_frame)
        self.screen = pygame.display.set_mode(self.px.get_rect()[2:])
        self.screen.blit(self.px, self.px.get_rect())

        return merged_image

    def crop_image(self, image_to_crop, output):
        cropped_image = image_to_crop[self.upper:self.lower, self.left:self.right]
        cv2.imwrite(output, cropped_image)

    def get_crop_commands(self, framerate):
        print("Preparing command to crop the video...")
        crop_command = ""

        crop_command += f'\nffmpeg \\\n'
        crop_command += f'-i {self.video} \\\n'
        crop_command += f'-filter:v "\\\n'
        crop_command += f'crop={self.right - self.left}:{self.lower - self.upper}:{self.left}:{self.upper}, \\\n'
        crop_command += f'scale={self.result_w}:{self.result_h}, \\\n'
        crop_command += f'fps={framerate}" \\\n'
        crop_command += f'-an \\\n'
        crop_command += f'{self.video.replace(".mp4", "_normalized.mp4")}\n'

        return crop_command


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        'video',
        type=str,
        help="Path to the video to be cropped.",
    )
    parser.add_argument(
        '-W', '--width',
        type=int,
        default=224,
        help="Width of the resulting video.",
    )
    parser.add_argument(
        '-H', '--height',
        type=int,
        default=224,
        help="height of the resulting video.",
    )
    parser.add_argument(
        '-f', '--framerate',
        type=int,
        default=20,
        help="height of the resulting video.",
    )
    parser.add_argument(
        '-s', '--script',
        action='store_true',
        help="When used, videos are not going to be cropped directly but script doing so is going to be created."
    )
    args = parser.parse_args()

    pygame.init()

    video_cropper = VideoCropper(args.video, args.width, args.height)
    image = video_cropper.get_image()
    video_cropper.set_crops()

    pygame.display.quit()

    video_cropper.recalculate_crops()
    command = video_cropper.get_crop_commands(args.framerate)
    if args.script:
        create_script(command)
    else:
        crop_video(command)
