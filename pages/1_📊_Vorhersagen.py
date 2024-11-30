"""
File containing all functions for showing the images 
with their predictions

File is written in pylint standard
"""

import os

import cv2
import numpy as np
import streamlit as st
import xml.etree.ElementTree as ET



def parse_xml(xml_file: str) -> list:
    """
    Parses XML-File for current image and extracts
    relevant data

    Parameters
    ----------
    xml_file : str
        Path to XML-File

    Returns
    -------
    list
    """

    tree = ET.parse(xml_file)
    root = tree.getroot()
    bboxes = []
    for member in root.findall('object'):
        label = member.find("name").text
        bbox = member.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bboxes.append((xmin, ymin, xmax, ymax, label))
        
    return bboxes


def draw_bboxes(image: np.ndarray, bboxes: list) -> np.ndarray:
    """
    Draws prediction of the image

    Parameters
    ----------
    image : np.ndarray
        Image where the prediction was made
    bboxes : list
        Read bboxes from the matching XML-File for the image

    Returns
    -------
    np.ndarray
    """
    for bbox in bboxes:
        xmin, ymin, xmax, ymax, label = bbox
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(image, label, (xmin, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return image


if 'index_ph' not in st.session_state:
    st.session_state.index_ph = 0


# -------Streamlit------- #
def main(directory: str) -> None:
    """
    Main visualization

    Parameters
    ----------
    directory : str
        Path to the directory where all the predictions are

    Returns
    -------
    None
    """

    files = sorted(os.listdir(directory))
    images = [file for file in files if file.lower().endswith((".png", ".jpg", ".jpeg")) and file.startswith("img")]

    sidebar = st.sidebar
    with sidebar:
        sidebar.header("Switch between images")
        col1, col2 = sidebar.columns(2)

        with col1:
            if st.button(":arrow_backward: Prev") and images:
                st.session_state.index_ph = (st.session_state.index_ph - 1) % len(images)
        with col2:
            if st.button("Next :arrow_forward:") and images:
                st.session_state.index_ph = (st.session_state.index_ph + 1) % len(images)

    if images:
        st.title(f"{images[st.session_state.index_ph]}")

        current_image = images[st.session_state.index_ph]
        image_path = os.path.join(directory, current_image)
        xml_path = os.path.join(directory, current_image.replace('.jpg', '.xml').replace('.png', '.xml').replace('.jpeg', '.xml'))

        if os.path.exists(xml_path):
            image = cv2.imread(image_path)
            bboxes = parse_xml(xml_path)
            image_with_bboxes = draw_bboxes(image, bboxes)
            image_with_bboxes = cv2.cvtColor(image_with_bboxes, cv2.COLOR_BGR2RGB)

            st.image(image_with_bboxes, caption=current_image, use_column_width=True)

        else:
            st.error(f"No XML file found for {current_image}")

    else:
        st.error("No predictions found in the directory")



if __name__ == "__main__":
    directory = "src/temp"
    main(directory=directory)
    