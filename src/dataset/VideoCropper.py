"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for cropping of video to a selected area.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 13. 02. 2022
Source: select crop area using pygame library
    (https://stackoverflow.com/questions/6136588/image-cropping-using-python/8696558)
"""

import sys
from pathlib import Path

import numpy as np
import cv2

from src.utils.params import parse_arguments
import src.utils.video_edit as video_edit

import contextlib

with contextlib.redirect_stdout(None):
    import pygame

COMPUTATIONAL_PRECISION = 1e6


class VideoCropper:
    def __init__(self, video_path, fps, result_w, result_h, scene_num, cam_num):
        self.video_path = Path(video_path)
        self.directory = self.video_path.parent

        self.result_fps = fps
        self.result_w = result_w
        self.result_h = result_h
        if result_w % 2 != 0:
            self.result_w += 1
            print(f"Resulting width has to be even, changed from {result_w} to {self.result_w}.", file=sys.stderr)
        if result_h % 2 != 0:
            self.result_h += 1
            print(f"Resulting height has to be even, changed from {result_h} to {self.result_h}.", file=sys.stderr)
        self.ratio = int(self.result_w / self.result_h * COMPUTATIONAL_PRECISION)

        self.scene_num = scene_num
        self.cam_num = cam_num

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

    def check_crop_ratio(self):
        print("- Checking if crop selection has a correct ratio.")

        crop_width = self.right - self.left
        crop_height = self.lower - self.upper
        crop_ratio = int((crop_width / crop_height) * COMPUTATIONAL_PRECISION)

        if crop_ratio > self.ratio:
            new_height = int((crop_width / self.ratio) * COMPUTATIONAL_PRECISION)
            print(f"-- Increasing height to {new_height} to fit the ratio.")
            half_difference = (new_height - crop_height) // 2
            self.upper -= half_difference
            self.lower += half_difference
            if (new_height - crop_height) % 2 != 0:
                self.lower += 1

        elif crop_ratio < self.ratio:
            new_width = int(crop_height * self.ratio / COMPUTATIONAL_PRECISION)
            print(f"-- Increasing width to {new_width} to fit the ratio.")
            half_difference = (new_width - crop_width) // 2
            self.left -= half_difference
            self.right += half_difference
            if (new_width - crop_width) % 2 != 0:
                self.right += 1

    def check_crop_size_to_result(self):
        print("- Checking if crop selection is large enough.")
        crop_size = self.right - self.left  # Width and height are now equal.
        if crop_size < self.result_w:  # Goal width and height are equal.
            print("-- Increasing crop selection size.")
            half_difference = (self.result_w - crop_size) // 2
            self.left -= half_difference
            self.right += half_difference
            self.upper -= half_difference
            self.lower += half_difference
            if (self.result_w - crop_size) % 2 != 0:
                self.right += 1
                self.lower += 1

    def check_crop_size_to_frame(self):
        print("- Checking if crop selection did not exceed the frame size.")
        crop_size = self.right - self.left  # Width and height are now equal.
        if crop_size > self.video_w or crop_size > self.video_h:
            difference = crop_size
            if crop_size > self.video_w:
                print("-- Width too high, decreasing both dimensions.")
                difference -= self.video_w
            if crop_size > self.video_h:
                print("-- Height too high, decreasing both dimensions.")
                difference -= self.video_h
            half_difference = difference // 2
            self.left += half_difference
            self.right -= half_difference
            self.upper += half_difference
            self.lower -= half_difference
            if difference % 2 != 0:
                self.right -= 1
                self.lower -= 1

    def check_crop_position(self):
        print("- Checking if crop selection is not out of the frame.")
        if self.left < 0:
            print("-- Moving crop to the right.")
            self.right -= self.left
            self.left = 0
        elif self.right > self.video_w:
            print("-- Moving crop to the left.")
            self.left -= self.right - self.video_w
            self.right = self.video_w
        if self.upper < 0:
            print("-- Moving crop down.")
            self.lower -= self.upper
            self.upper = 0
        elif self.lower > self.video_h:
            print("-- Moving crop up.")
            self.upper -= self.lower - self.video_h
            self.lower = self.video_h

    def recalculate_crops(self):
        print("\nRecalculating crop area...")
        print("- Video information")
        print(f"-- Frame width - {self.video_w}")
        print(f"-- Frame height - {self.video_h}")
        print(f"-- Ratio * {COMPUTATIONAL_PRECISION}- {self.ratio}")
        print("- Original crop coordinates")
        print(f"-- Left - {self.left}")
        print(f"-- Right - {self.right}")
        print(f"-- Upper - {self.upper}")
        print(f"-- Lower - {self.lower}")

        self.check_crop_direction()
        self.check_crop_ratio()
        self.check_crop_size_to_result()
        self.check_crop_size_to_frame()
        self.check_crop_position()

        print("- Recalculated crop coordinates")
        print(f"-- Left - {self.left}")
        print(f"-- Right - {self.right}")
        print(f"-- Upper - {self.upper}")
        print(f"-- Lower - {self.lower}")

    def get_image(self, image_count=10):
        print("Merging video to single image...")
        image_channels = 3

        video = cv2.VideoCapture(f'{self.video_path}')
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
                print(f"-- Frame from video could not be obtained.", file=sys.stderr)
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

    def get_crop_commands(self):
        """
        ffmpeg command to crop a video:
            ffmpeg
            -i input_file
            -filter:v "crop=width:height:start_width:start_height, scale=width:height, fps=fps"
            -an
            output_file

        :return: String with constructed ffmpeg command to crop and scale the video.
        """
        print("\nPreparing command to crop the video...")
        crop_command = ""

        crop_command += f'\nffmpeg \\\n'
        crop_command += f'-i {self.video_path} \\\n'
        crop_command += f'-filter:v "\\\n'
        crop_command += f'crop={self.right - self.left}:{self.lower - self.upper}:{self.left}:{self.upper}, \\\n'
        crop_command += f'scale={self.result_w}:{self.result_h}, \\\n'
        crop_command += f'fps={self.result_fps}" \\\n'
        crop_command += f'-an \\\n'
        crop_command += f'{self.directory}/scene{self.scene_num:03d}_cam{self.cam_num}_normalized.mp4\n'

        return crop_command


def main():
    args = parse_arguments()

    pygame.init()

    video_cropper = VideoCropper(args.location, args.framerate, args.width, args.height, args.scene_num, args.cam_num)
    _ = video_cropper.get_image()
    video_cropper.set_crops()

    pygame.display.quit()

    video_cropper.recalculate_crops()

    command = video_cropper.get_crop_commands()
    if args.script:
        video_edit.create_script(command)
        print("- Crop script created.")
    else:
        video_edit.execute_command(command)
        print("- Video cropped.")


if __name__ == "__main__":
    main()
