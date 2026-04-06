# src/prepare_data.py

import os
import xml.etree.ElementTree as ET
from PIL import Image


RAW_DIR = "data/raw/NEU-DET/train"
IMG_DIR = os.path.join(RAW_DIR, "images")
ANN_DIR = os.path.join(RAW_DIR, "annotations")

OUTPUT_DIR = "data/processed"


def find_image(img_filename):
    """
    Search for image inside class subfolders
    """

    name = os.path.splitext(img_filename)[0]

    for root, _, files in os.walk(IMG_DIR):
        for file in files:
            if os.path.splitext(file)[0] == name:
                return os.path.join(root, file)

    return None


def prepare_dataset():

    print("Preparing classification dataset...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    missing_count = 0

    for xml_file in os.listdir(ANN_DIR):

        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(ANN_DIR, xml_file)

        tree = ET.parse(xml_path)
        root = tree.getroot()

        img_filename = root.find("filename").text
        img_path = find_image(img_filename)

        if img_path is None:
            missing_count += 1
            print(f" Missing image: {img_filename}")
            continue

        image = Image.open(img_path)

        for i, obj in enumerate(root.findall("object")):

            label = obj.find("name").text

            bbox = obj.find("bndbox")
            xmin = int(float(bbox.find("xmin").text))
            ymin = int(float(bbox.find("ymin").text))
            xmax = int(float(bbox.find("xmax").text))
            ymax = int(float(bbox.find("ymax").text))

            cropped = image.crop((xmin, ymin, xmax, ymax))

            class_dir = os.path.join(OUTPUT_DIR, label)
            os.makedirs(class_dir, exist_ok=True)

            save_name = f"{os.path.splitext(img_filename)[0]}_{i}.jpg"
            save_path = os.path.join(class_dir, save_name)

            cropped.save(save_path)

    print("\n Dataset prepared at:", OUTPUT_DIR)
    print(f"Missing images: {missing_count}")


if __name__ == "__main__":
    prepare_dataset()