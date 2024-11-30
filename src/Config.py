"""
This Module holds all config-specific paths
and variables

File is written in pylint standard
"""

import os
import logger as log



class Config:
    """
    Class defining configs and paths for code
    
    ...

    Methods
    -------
    folder_res:
        Path to the folder 'res'
    folder_src:
        Path to the folder 'src'
    folder_test:
        Path to the folder 'test'

    Private Methods
    ---------------
    __path_checker:
        Checks if the path exists
    """

    def __init__(self):
        self.__logger = log.get_logger()
        self.defaultpaths = os.listdir()

    def folder_res(self) -> str:
        """
        Path to the folder 'res'

        Returns
        -------
        str
        """

        if "res" in self.defaultpaths:
            res_folder = os.path.join(".", "res")
        else:
            res_folder = os.path.join("..", "res")
        return self.__path_checker(res_folder)

    def folder_src(self) -> str:
        """
        Path to the folder 'src'
        
        Returns
        -------
        str
        """

        if "src" in self.defaultpaths:
            src_folder = os.path.join(".", "src")
        else:
            src_folder = os.path.join("..", "src")
        return self.__path_checker(src_folder)

    def folder_test(self) -> str:
        """
        Path to the folder 'test'
        
        Returns
        -------
        str
        """

        if "test" in self.defaultpaths:
            test_folder = os.path.join(".", "test")
        else:
            test_folder = os.path.join("..", "test")
        return self.__path_checker(test_folder)


    #----------Private Methods----------#
    def __path_checker(self, path: str) -> str:
        """
        Checks if the path exists
  
        Parameters
        ----------
        path : str
            Path to the folder

        Returns
        -------
        str
        """

        return path if os.path.exists(path) else self.__logger.error("Path does not exist.")



if __name__ == "__main__":
    config = Config(logger=log.get_logger()).folder_src()
