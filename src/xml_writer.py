"""
This module provides functionality to generate XML annotations for images based on predictions
from ./res/ssd_mobilenet_v2_20k

Usage:
    python ./src/semi_supervised_learning.py --path "path_to_data"

Args:
    --path (str): Path to the image path to make predictions

Example:
    python ./src/semi_supervised_learning.py --path "/path/to/data"

File is written in pylint standard
"""

import os
import glob
import argparse
from typing import Tuple
import xml.etree.cElementTree as ET

import cv2

import logger as log



def write_xml_annotations(predictions: dict, output_file: str) -> None:
    """
    Writes xml file

    Parameters
    ----------
    predictions: dict
        Prediction Directory which is written into the XML-File
    output_file: str
        How to outputfile should be named

    Returns
    -------
    None
    """
    root = ET.Element("annotation")
    folder = ET.SubElement(root, "folder")
    folder.text = "predicted_images"
    filename = ET.SubElement(root, "filename")
    filename.text = predictions['filename']
    path = ET.SubElement(root, "path")
    path.text = predictions['path']
    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"
    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    width.text = str(predictions['width'])
    height = ET.SubElement(size, "height")
    height.text = str(predictions['height'])
    depth = ET.SubElement(size, "depth")
    depth.text = str(predictions['depth'])
    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"

    for obj in predictions['objects']:
        object_elem = ET.SubElement(root, "object")
        name = ET.SubElement(object_elem, "name")
        name.text = obj['name']
        pose = ET.SubElement(object_elem, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(object_elem, "truncated")
        truncated.text = "0"
        difficult = ET.SubElement(object_elem, "difficult")
        difficult.text = "0"
        bndbox = ET.SubElement(object_elem, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(obj['xmin'])
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(obj['ymin'])
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(obj['xmax'])
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(obj['ymax'])

    tree = ET.ElementTree(root)
    tree.write(output_file)

    log.get_logger().info("XML-File was written successfully")


def process_prediction_data(image: str, detection: dict =None, confidence: float =0.25) -> Tuple[dict, str]:
    """
    Writes dictionary to create a XML-File

    Parameters
    ----------
    image : str
        filepath -> Used to extract filename
    detection : dict (default=None)
        Detection directory from the detection model
    confidence : float (default=0.25)
        pass

    Returns
    -------
    dict, str
    """
    image = image.replace("\\", "/")

    filename = str(image).rsplit('/', maxsplit=1)[-1]
    path = os.path.abspath(image)

    image_read = cv2.imread(image)
    h, w, _ = image_read.shape
    
    if detection is not None:
        detections = detection
    else:
        detections, _ = ps.detection(image=image_read)

    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(int)
 
    classes_dict = {
        1 : "table",
        2 : "total-price"
    }

    objects = list()
    for box, score, class_id in zip(boxes, scores, classes):
        # box = ymin, xmin, ymax, xmax
        if score > confidence:
            label = classes_dict[class_id]

            ymin, xmin, ymax, xmax = box
            ymin = int(ymin * h)
            xmin = int(xmin * w)
            ymax = int(ymax * h)
            xmax = int(xmax * w)

            objects.append({"name": label, "xmin":xmin, "ymin":ymin, "xmax":xmax, "ymax":ymax})

    predictions = {
        "filename" : filename,
        "path" : path,
        "width" : w,
        "height" : h,
        "depth" : 1,
        "objects" : objects
    }

    return predictions, filename



if __name__ == "__main__":
    from PredictionService import PredictionService
    parser = argparse.ArgumentParser(description="Get path to unlabeled data.")
    parser.add_argument("--path", required=True, help="Get path to unlabeled data.")
    args = parser.parse_args()

    ps = PredictionService()
    for img in glob.glob(pathname=f"{args.path}/*.png"):
        pred, f_name = process_prediction_data(image=img, confidence=0.25)

        output_file = f"{args.path}/{f_name.split('.')[0]}.xml"
        write_xml_annotations(pred, output_file)
        print(f"XML annotations written to {output_file}")
