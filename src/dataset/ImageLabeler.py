"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for labeling images.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 11. 04. 2022
"""

from pathlib import Path
import pickle

import cv2

from src.utils.params import parse_arguments


def import_classes(classes_location):
    if classes_location is not None:
        with open(classes_location, 'rb') as f:
            classes = pickle.load(f)
    else:
        classes = {32: "unsorted"}

    return classes


class ImageLabeler:
    def __init__(self, directory, classes_location):
        self.directory = Path(directory)
        self.image_paths = Path(directory).glob('*.png')
        self.classes = import_classes(classes_location)

        print("Image Labeler (IM) initialized.")

    def label(self):
        print("IM - Labeling images...")
        print("IM -- \"Esc\" - quit labeling.")
        self.print_classes()

        for image_path in self.image_paths:
            image = cv2.imread(str(image_path))
            cv2.namedWindow("Image")
            cv2.imshow("Image", image)
            k = cv2.waitKey(0) & 0xff
            if k == 27:
                break
            elif k not in self.classes:
                class_name = input(f"IM --- Key: \"{chr(k)}\" = Class: ")
                self.classes[k] = class_name

            dst_dir = self.directory / self.classes[k]
            dst = dst_dir / image_path.name
            Path(dst_dir).mkdir(parents=True, exist_ok=True)
            image_path.replace(dst)

        cv2.destroyWindow("Image")

    def edit(self):
        self.print_classes()

        k = ord(input(f"Enter the key of class name you want to change (or add): "))
        c = input(f"Enter the new class name (or nothing to delete): ")

        if c == "":
            for image_path in (self.directory / self.classes[k]).glob("*.*"):
                src = Path(image_path)
                dst = self.directory / src.name
                src.replace(dst)
            (self.directory / self.classes[k]).rmdir()
            del self.classes[k]
        else:
            (self.directory / self.classes[k]).rename(self.directory / c)
            self.classes[k] = c

    def export_classes(self):
        file_name = self.directory / 'classes.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(self.classes, file)

    def print_classes(self):
        print("IM -- Available classes:")
        for k, c in self.classes.items():
            print(f"IM --- Key: \"{chr(k)}\" = Class: {c}")


def main():
    args = parse_arguments()

    image_labeler = ImageLabeler(args.location, args.classes_location)

    while True:
        print("IM - Choose your task.")
        print("IM -- Enter \"1\" to label images.")
        print("IM -- Enter \"2\" to edit labels.")
        print("IM -- Enter \"0\" to exit.")

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

        print("IM - Task finished.")


if __name__ == "__main__":
    main()
