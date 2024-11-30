""" 
In this file everything about the prediction service
will happen, including OCR and Detection

File is written in pylint standard
"""

import time

import cv2
import easyocr
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import logger as log
from xml_writer import *
from Config import Config
from ImageProcessing import ImageProcessing



class PredictionService(Config):
    """
    Class holding all methods for OCR and Object-Detection

    ...

    Attributes
    ----------
    confidence : float (default=0.4)
        Model-Confidence where data should be collected
    write_xml : bool (default = True)
        If True -> Creates XML-File with the predictions for a given
        image

    Methods
    -------
    detection:
        Detects the two classes (table, total_price) on
        the given image. Also writes xml_file if self.write_xml
    extract_detection:
        Extracts the detected objects from the image 
        for postprocessing. If visualize = True it
        shows the detected BBoxes on the image
    extract_text:
        Uses OCR to extract text from image

    Private Methods
    ---------------
    __filename_splitter:
        Gets the filename from absolute path
    __visualize
        Visualizes detections
    """

    def __init__(self, confidence: float =0.4, write_xml: bool =True):
        super().__init__()
        self.confidence = confidence
        self.write_xml = write_xml
        self.model = tf.saved_model.load(f"{self.folder_res()}/model/saved_model/")
        self.ocr_reader = easyocr.Reader(["en", "de"])
        self.__logger = log.get_logger()


    def detection(self, img: str) -> dict:
        """ 
        Detects the two classes (table, total_price) on the given image.
        Also writes xml_file if self.write_xml

        Parameters
        ----------
        img : str
            Imagepath where objects should be detected from

        Returns
        -------
        dict
        """

        orig_image = ImageProcessing.model_preprocess(img=img)
        start_time = time.time()
        detection = self.model(orig_image)
        end_time = time.time()
        self.__logger.info(f"A detection was made in {round(end_time-start_time, 4)} sec.")

        if self.write_xml:
            pred, f_name = process_prediction_data(image=img,
                                                   detection=detection,
                                                   confidence=self.confidence)
            output_file = f"{self.folder_src()}/temp/{f_name.split('.')[0]}.xml"
            write_xml_annotations(predictions=pred, output_file=output_file)

        return detection
            

    def extract_detection(self, img: str, visualize: bool = False) -> None:
        """ 
        Extracts the detected objects from the image for postprocessing.
        If visualize = True it shows the detected BBoxes on the image

        Parameters
        ----------
        img : str
            Needed for detection and filename parsing
        visualize : bool (default=False)
            Visualizes detection if set to True
            Should ONLY be used for debugging, NOT in production

        Returns
        -------
        None
        """

        detections = self.detection(img=img)

        orig_image = cv2.imread(img)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(int)

        # Should only be used for debugging
        if visualize:
            self.__visualize(image=image,
                                boxes=boxes,
                                scores=scores,
                                classes=classes)
            
        for box, score, class_id in zip(boxes, scores, classes):
            if score > self.confidence:
                ymin, xmin, ymax, xmax = box
                ymin = int(ymin * image.shape[0])
                xmin = int(xmin * image.shape[1])
                ymax = int(ymax * image.shape[0])
                xmax = int(xmax * image.shape[1])

                img_cut = orig_image[ymin:ymax, xmin:xmax]

                if class_id == 1:
                    plt.imsave(
                        f"{self.folder_src()}/temp/table_{self.__filename_splitter(img)}.png", img_cut
                        )

                else:
                    img_cut = ImageProcessing(img=img_cut).add_border()
                    plt.imsave(
                        f"{self.folder_src()}/temp/price_{self.__filename_splitter(img)}.png", img_cut
                        )


    def extract_text(self, img: str) -> float:
        """ 
        Uses OCR to extract text from image

        Parameters
        ----------
        img : str
            Imagepath to the image where text should be extracted

        Returns
        -------
        float
        """

        total: float = 0
        replace_dict = {
            "[": "",
            "]": "",
            "," : ".",
            " " : "",
            "_" : "",
            "-" : "",
            "/" : ""
        }

        # rotation_info=[90, 180, 270] for better detection?
        start_time = time.time()
        extracted_text = self.ocr_reader.readtext(
            f"{self.folder_src()}/temp/{img}", detail=0, batch_size = 12
            )[0]
        end_time = time.time()

        for key, value in replace_dict.items():
            if key in extracted_text:
                extracted_text = extracted_text.replace(key, value)
        total = float(extracted_text)

        self.__logger.info(f"Text was extracted in {round(end_time-start_time, 4)} sec.")
        return total
    

    #----------Private Methods----------#
    def __filename_splitter(self, img: str) -> str:
        """ 
        Gets the filename from absolute path

        Parameters
        ----------
        img : str
            Absolute path

        Returns
        -------
        str
        """

        return str(img).split("/")[-1].split(".")[0]
    
    def __visualize(self, image: np.ndarray, boxes: list, scores: list, classes: list) -> None:
        """ 
        Visualizes detections

        Parameters
        ----------
        image : np.ndarray
            Image where detections should be drawn on and visualized
        boxes : list
            Coordinates of the detected objects
        scores : list
            Confidence with which the objects were detected
        classes : list
            Classes of the detected objects

        Returns
        -------
        None
        """
        
        for box, score, class_id in zip(boxes, scores, classes):
            if score > self.confidence:
                ymin, xmin, ymax, xmax = box
                ymin = int(ymin * image.shape[0])
                xmin = int(xmin * image.shape[1])
                ymax = int(ymax * image.shape[0])
                xmax = int(xmax * image.shape[1])

                color = (0, 255, 0)
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness=2)

                if class_id == 1:
                    label = f'Table {score:.2f}'
                else:
                    label = f'total_price {score:.2f}'

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                text_origin = (xmin, ymin - 10)
                image = cv2.putText(
                    image,
                    label,
                    text_origin,
                    font,
                    font_scale,
                    color,
                    font_thickness,
                    cv2.LINE_AA
                    )

        plt.subplots(figsize=(20, 15))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

        return None



if __name__ == "__main__":
    import glob

    ps = PredictionService(confidence=0.3, write_xml=False)
    files = glob.glob("./res/data/*.JPEG")
    files = [str(file).replace("\\", "/") for file in files]

    for file in files:
        ps.extract_detection(img=file, visualize=True)
