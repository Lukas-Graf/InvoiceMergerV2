"""
Module containing a class (ImageProcessing) with
all necessary processing steps for model input 
|-> Detection and OCR <-|

File is written in pylint standard
"""

from typing import List, Any

import cv2
import imutils
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import logger as log


class ImageProcessing:
    """
    Class defining all the ImageProcessing methods for Model Prediction and OCR
    
    ...

    Attributes
    ----------
    img : Any (str, np.ndarray)
        Image which should be operated on
    visualize : bool (default = False)
        Parameter to visualize processed image

    Methods
    -------
    show_plot (staticmethod):
        Visualizes the image as a plot
    model_preprocess (staticmethod):
        Prepares image for detection model
    invert_image:
        Invertes given image
    binarize_image:
        Binarizes given image in a defined area (thresh <= col_channel <= max_val)
    noise_removal:
        Removes noise from given image
    change_font:
        Changes the font of a given image to "thick" or "thin"
    rmv_border:
        Removes border of a given image
    add_border:
        Adds border to given image
    rotate_image -> Not implemented yet:
        Rotates the image so it is straight

    Private Methods
    ---------------
    __grayscale:
        Changes color channels from bgr to gray
    __show_img:
        Shows given image
    """

    def __init__(self, img: Any, visualize: bool = False):
        self.visualize = visualize
        self.__logger = log.get_logger()

        if isinstance(img, str):
            self.img = cv2.imread(img)
        elif isinstance(img, np.ndarray):
            self.img = img
        else:
            self.__logger.error(f"Img has to be of type str or np.ndarray not {type(img)}")
            raise TypeError(f"Img has to be of type str or np.ndarray not {type(img)}")

    @staticmethod
    def show_plot(img: str, method: str = "viridis") -> None:
        """
        Visualizes the image as a plot

        Parameters
        ----------
        img : Any (np.ndarray or str) (default="viridis")
            Image which should be visualized
        method : str
            Method how the image should be visualized

        Returns
        -------
        None
        """

        plt.subplots(figsize=(12, 8))

        if isinstance(img, str):
            img = cv2.imread(img)
        else:
            raise TypeError(f"Img has to be of type str or np.ndarray not {type(img)}")

        plt.imshow(img, cmap=str(method))
        plt.show()

        return None

    @staticmethod
    def model_preprocess(img: Any) -> tf.Tensor:
        """
        Prepares image for detection model

        Parameters
        ----------
        img: Any (np.ndarray or str)
            Image which be preprocessed

        Returns
        -------
        tf.Tensor
        """

        if isinstance(img, str):
            img = cv2.imread(img)
        else:
            raise TypeError(f"Img has to be of type str or np.ndarray not {type(img)}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = tf.convert_to_tensor(img, dtype=tf.uint8)
        img_tensor_dim = tf.expand_dims(img_tensor, axis=0)

        log.get_logger().info("Image was successfully preprocessed")
        return img_tensor_dim

    def invert_image(self) -> np.ndarray:
        """
        Invertes given image

        Returns
        -------
        np.ndarray
        """

        inverted_img = cv2.bitwise_not(self.img)

        if self.visualize:
            self.__show_img(inverted_img, "Inverted Image")

        self.__logger.info("Image was inverted successfully")
        return inverted_img

    def binarize_image(self, thresh: int = 110, max_val: int = 230) -> np.ndarray:
        """
        Binarizes given image in a defined area (thresh <= color_channel <= max_val)

        Parameters
        ----------
        thresh : int (default = 110)
            Pixel values below this threshold are set to black (0)
        max_val : int (default = 230)
            Pixel values above this threshold are set to white (255)

        Returns
        -------
        np.ndarray
        """

        img_grayscale = self.__grayscale(self.img)
        thresh, img_bw = cv2.threshold(img_grayscale,
                                       thresh=thresh,
                                       maxval=max_val,
                                       type=cv2.THRESH_BINARY)

        if self.visualize:
            self.__show_img(img_bw, "ImageBlackWhite")

        self.__logger.info("Image was binarized successfully")
        return img_bw

    def noise_removal(self) -> np.ndarray:
        """
        Removes noise from given image

        Returns
        -------
        np.ndarray
        """

        kernel = np.ones((1, 1), np.uint8)
        img_dilated = cv2.dilate(self.img, kernel=kernel, iterations=1)

        kernel = np.ones((1, 1), np.uint8)
        img_erode = cv2.erode(img_dilated, kernel=kernel, iterations=1)

        img_morph = cv2.morphologyEx(img_erode, cv2.MORPH_CLOSE, kernel=kernel)
        img_blur = cv2.medianBlur(img_morph, 1)

        if self.visualize:
            self.__show_img(img_blur, "ImageNoNoise")

        self.__logger.info("Image-Noise was removed successfully")
        return img_blur

    def change_font(self, mode: str = "thick") -> np.ndarray:
        """
        Changes the font of a given image to 'thick' or 'thin'

        Parameters
        ----------
        mode : str (default="thick")
            Makes font thicker or thinner

        Returns
        -------
        np.ndarray
        """

        img_inverted = cv2.bitwise_not(self.img)
        kernel = np.ones((2, 2), np.uint8)

        if mode == "thick":
            img_dilated = cv2.dilate(img_inverted, kernel=kernel, iterations=1)
            img = cv2.bitwise_not(img_dilated)
        elif mode == "thin":
            img_eraded = cv2.erode(img_inverted, kernel=kernel, iterations=1)
            img = cv2.bitwise_not(img_eraded)
        else:
            raise ValueError(f"Given value ({mode}) for variable 'mode' is not known.")

        if self.visualize:
            self.__show_img(img, f"Image{(mode).upper()}")

        self.__logger.info("Font was changed successfully")
        return img

    def rmv_border(self) -> np.ndarray:
        """
        Removes border of a given image

        Returns
        -------
        np.ndarray
        """

        img = self.img
        img_grayscale = self.__grayscale(img)
        contours, _ = cv2.findContours(img_grayscale,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cnt_sorted = sorted(contours, key=lambda x: cv2.contourArea(x))
        cnt = cnt_sorted[-1]
        x, y, w, h = cv2.boundingRect(cnt)
        crop = img[y:y + h, x:x + w]

        if self.visualize:
            self.__show_img(crop, "ImageNoBorder")

        self.__logger.info("Border was removed successfully")
        return crop

    def add_border(self, color: List[int] = [255] * 3, measures: List[int] = [150] * 4) -> np.ndarray:
        """
        Adds border to given image

        Parameters
        ----------
        color : list (default = [255, 255, 255])
            Color of the border which should be added
        measures : list (default = [150, 150, 150, 150])
            Thicknes of the border which should be added

        Returns
        -------
        np.ndarray
        """

        top, bottom, left, right = measures
        img_border = cv2.copyMakeBorder(self.img,
                                        top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=color)

        if self.visualize:
            self.__show_img(img_border, "ImageWithBorder")

        self.__logger.info("Border was added successfully")
        return img_border

    def rotate_image(self) -> np.ndarray:
        pass

    # ----------Private Methods----------#
    def __grayscale(self, img: np.ndarray) -> np.ndarray:
        """
        Changes color channels from bgr to gray

        Parameters
        ----------
        img : np.ndarray
            Image which should be turned gray

        Returns
        -------
        np.ndarray
        """

        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def __show_img(self, img: np.ndarray, img_name: str) -> None:
        """
        Shows given Image

        Parameters
        ----------
        img: np.ndarray
        img_name: str

        Returns
        -------
        np.ndarray
        """

        resized = imutils.resize(img, width=600)
        cv2.imshow(img_name, resized)
        cv2.waitKey(0)
        return None


if __name__ == "__main__":
    # Visualize=True does not work in colab use plt to show plot
    TEST_IMG_PATH = r"test\temp\Receipt_2023-12-23_001844.jpg"
    img_pre = ImageProcessing(img=TEST_IMG_PATH, visualize=True)
    img_pre.add_border()
