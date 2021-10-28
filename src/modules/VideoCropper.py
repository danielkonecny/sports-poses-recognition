"""
Source: https://stackoverflow.com/questions/6136588/image-cropping-using-python/8696558
"""


from argparse import ArgumentParser
from PIL import Image

import contextlib
with contextlib.redirect_stdout(None):
    import pygame


# TODO - Import video instead of image and create image from multiple frames merged together with some opacity.
# TODO - Recalculate selected rectangle to square with the center staying at the same place.
# TODO - Check the size of the square to be big enough and increase the size if needed.
# TODO - Create and run ffmpeg command to crop, scale and other operations.
# TODO - Make available to run on multiple videos as a module.


class VideoCropper:
    def __init__(self, image):
        self.image = image
        self.output = '../dip-data/task/out.png'
        pygame.init()
        self.px = pygame.image.load(self.image)
        self.screen = pygame.display.set_mode(self.px.get_rect()[2:])
        self.screen.blit(self.px, self.px.get_rect())

    def display_image(self, top_left, prior):
        # ensure that the rect always has positive width, height
        x, y = top_left
        width = pygame.mouse.get_pos()[0] - top_left[0]
        height = pygame.mouse.get_pos()[1] - top_left[1]
        if width < 0:
            x += width
            width = abs(width)
        if height < 0:
            y += height
            height = abs(height)

        # eliminate redundant drawing cycles (when mouse isn't moving)
        current = x, y, width, height
        if not (width and height):
            return current
        if current == prior:
            return current

        # draw transparent box and blit it onto canvas
        self.screen.blit(self.px, self.px.get_rect())
        display_im = pygame.Surface((width, height))
        display_im.fill((128, 128, 128))
        pygame.draw.rect(display_im, (32, 32, 32), display_im.get_rect(), 1)
        display_im.set_alpha(128)
        self.screen.blit(display_im, (x, y))
        pygame.display.flip()

        # return current box extents
        return x, y, width, height

    def main_loop(self):
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
        return top_left + bottom_right

    def crop(self):
        pygame.display.flip()

        left, upper, right, lower = self.main_loop()

        # ensure output rect always has positive width, height
        if right < left:
            left, right = right, left
        if lower < upper:
            lower, upper = upper, lower
        im = Image.open(self.image)
        im = im.crop((left, upper, right, lower))
        pygame.display.quit()
        im.save(self.output)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        'input',
        type=str,
        help="Path to the video to be cropped.",
    )
    args = parser.parse_args()

    video_cropper = VideoCropper(args.input)

    video_cropper.crop()
