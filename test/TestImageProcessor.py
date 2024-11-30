""" 
Script tests all the methods in the ImageProcessing
class
"""

import sys

import unittest
import numpy as np

sys.path.append("./src")
from Config import Config
from ImageProcessing import ImageProcessing



class TestImageProcessor(unittest.TestCase):
    """ 
    Class defining all the methods for testing ImageProcessing Class
    ---------------------
    Methods:
        setUp (staticmethod)
            Set up the class
        test_invert_image (staticmethod)
            Tests method invert_image from ImageProcessing
        test_binarize_image
            Tests method binarize_image from ImageProcessing
        test_noise_removal
            Tests method noise_removal from ImageProcessing
        test_change_font
            Tests method change_font from ImageProcessing
        test_rmv_border
            Tests method rmv_border from ImageProcessing
        test_add_border
            Tests method add_border from ImageProcessing
    """

    def setUp(self):
        """ 
        Set up the class
        """
        self.preprocessor = ImageProcessing(
            img=f"{Config().folder_test()}/test_image.jpg"
            )

    def test_invert_image(self):
        """ 
        Tests method invert_image from ImageProcessing
        """
        result = self.preprocessor.invert_image()
        self.assertIsInstance(
            result,
            np.ndarray,
            "Methods 'invert_image' return value is not a NumPy array")     

    def test_binarize_image(self):
        """ 
        Tests method binarize_image from ImageProcessing
        """
        result = self.preprocessor.binarize_image()
        self.assertIsInstance(
            result,
            np.ndarray,
            "Methods 'binarize_image' return value is not a NumPy array")

    def test_noise_removal(self):
        """ 
        Tests method noise_removal from ImageProcessing
        """
        result = self.preprocessor.noise_removal()
        self.assertIsInstance(
            result,
            np.ndarray,
            "Methods 'noise_removal' return value is not a NumPy array")

    def test_change_font(self):
        """ 
        Tests method change_font from ImageProcessing
        """
        result = self.preprocessor.change_font()
        self.assertIsInstance(
            result,
            np.ndarray,
            "Methods 'change_font' return value is not a NumPy array")

    def test_rmv_border(self):
        """ 
        Tests method rmv_border from ImageProcessing
        """
        result = self.preprocessor.rmv_border()
        self.assertIsInstance(
            result,
            np.ndarray,
            "Methods 'rmv_border' return value is not a NumPy array")

    def test_add_border(self):
        """ 
        Tests method add_border from ImageProcessing
        """
        result = self.preprocessor.add_border()
        self.assertIsInstance(
            result,
            np.ndarray,
            "Methods 'add_border' return value is not a NumPy array")



if __name__ == '__main__':
    unittest.main()
