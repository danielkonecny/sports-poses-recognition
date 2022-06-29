"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for labeling images.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 28. 06. 2022
"""

from argparse import ArgumentParser
from pathlib import Path
import pickle

import cv2

CAM_COUNT = 3
CAM_SHOWN = 1


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        'dir',
        type=str,
        help="Directory of the (un)sorted images. Sorted ones in their directories named by their classes. "
             "Unsorted in a directory named \"sort\"."
    )
    parser.add_argument(
        '-c', '--classes',
        type=str,
        default=None,
        help="Path to the dictionary with classes saved with pickle."
    )
    return parser.parse_args()


class ImageLabeler:
    def __init__(self, directory, classes):
        self.directory = Path(directory)
        self.image_paths = (self.directory / "sort").rglob(f'*/cam{CAM_SHOWN}/*.png')
        self.classes = self.import_classes(classes)

        print("Image Labeler (IL) initialized.")

    def label(self):
        print("IL - Labeling images...")
        print("IL -- \"Esc\" - quit labeling.")
        self.print_classes()

        cv2.namedWindow("Image")

        for image_path in self.image_paths:
            image = cv2.imread(str(image_path))
            cv2.imshow("Image", image)
            k1 = cv2.waitKey(0) & 0xff
            cv2.imshow("Image", image)
            k2 = cv2.waitKey(0) & 0xff

            if k1 == 27 or k2 == 27:
                break

            k = chr(k1) + chr(k2)
            if k not in self.classes:
                while True:
                    class_name = input(f"IL --- Key: \"{k}\" = Class: ")
                    if class_name == "sort":
                        print(f"IL --- Cannot name a class \"sort\", that is a default name for directory "
                              f"of unsorted images.")
                    else:
                        self.classes[k] = class_name
                        break

            # TODO - Make more general for other directory structures than scene/cam/img.png.
            for cam_index in range(0, CAM_COUNT):
                new_image_name = image_path.name.replace(f"cam{CAM_SHOWN}", f"cam{cam_index}")

                dst_dir = self.directory / self.classes[k]
                dst = dst_dir / new_image_name
                Path(dst_dir).mkdir(parents=True, exist_ok=True)

                general_image_path = image_path.parents[1] / f"cam{cam_index}" / new_image_name
                general_image_path.replace(dst)

        cv2.destroyWindow("Image")

    def edit(self):
        self.print_classes()

        k = input(f"Enter the key of class name you want to change (or add): ")
        c = input(f"Enter the new class name (or nothing to delete): ")

        if c == "":
            for image_path in (self.directory / self.classes[k]).glob("*.*"):
                src = Path(image_path)
                dst = self.directory / src.name
                src.replace(dst)
            (self.directory / self.classes[k]).rmdir()
            del self.classes[k]
        else:
            if k in self.classes:
                dst_dir = self.directory / self.classes[k]
                dst_dir.rename(self.directory / c)
                self.classes[k] = c
            else:
                self.classes[k] = c
                dst_dir = self.directory / self.classes[k]
                dst_dir.mkdir()

    @staticmethod
    def import_classes(classes_location):
        if classes_location is not None:
            with open(classes_location, 'rb') as f:
                classes = pickle.load(f)
        else:
            classes = {"  ": "unsorted"}

        return classes

    def export_classes(self):
        file_name = self.directory / 'classes.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(self.classes, file)

    def print_classes(self):
        print("IL -- Available classes:")
        for k, c in self.classes.items():
            print(f"IL --- Key: \"{k}\" = Class: {c}")


def main():
    args = parse_arguments()

    image_labeler = ImageLabeler(args.dir, args.classes)

    while True:
        print("IL - Choose your task.")
        print("IL -- Enter \"1\" to label images.")
        print("IL -- Enter \"2\" to edit labels.")
        print("IL -- Enter \"0\" to exit.")

        k = input()

        if k == '1':
            image_labeler.label()
        elif k == '2':
            image_labeler.edit()
        elif k == '0':
            break
        else:
            continue

        image_labeler.export_classes()

        print("IL - Task finished.")


if __name__ == "__main__":
    main()
