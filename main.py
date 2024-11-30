""" 
File containing all functions for the main multi-
page app

File is written in pylint standard
"""

import os
import sys
import time
import requests
from PIL import Image
from typing import NoReturn

import streamlit as st

sys.path.append("./src")
import logger as log
from Config import Config
from Invoice import Invoice
from PredictionService import PredictionService



if "config" not in st.session_state:
    st.session_state.config = Config()

if "invoice" not in st.session_state:
    st.session_state.invoice = Invoice()

@st.cache_resource
def load_detection_model() -> PredictionService:
    """
    Loads PredictionsService class

    Returns
    -------
    PredictionsService (Instance)
    """
    return PredictionService(confidence=0.3)

@st.cache_data
def get_roi(uploaded_images: list) -> None:
    """ 
    Cleans Image-Folder, Saves uploaded images and extracts detections
    from it

    Parameters
    ----------
    uploaded_images : list
        Images which were uploaded from the user

    Returns
    -------
    None
    """

    img_length: int = len(uploaded_images)
    if img_length == 1:
        log.get_logger().info(f"{img_length} invoice was uploaded")
    else:
        log.get_logger().info(f"{img_length} invoices were uploaded")

    #Delete old files
    for file in os.listdir(f"{st.session_state.config.folder_src()}/temp/"):
        os.remove(f"src/temp/{file}")

    for idx, file in enumerate(uploaded_images):
        img = Image.open(file)
        img = img.save(f"src/temp/img_{idx}.jpg")

    if uploaded_images is not None:
        for img in os.listdir(f"{st.session_state.config.folder_src()}/temp/"):
            detection_model.extract_detection(
                img=f"{st.session_state.config.folder_src()}/temp/{img}"
                )

    return None


def get_total_price() -> float:
    """ 
    Sums up the total prices of all the invoices

    Returns
    -------
    float
    """

    total: float = 0
    for img in os.listdir(f"{st.session_state.config.folder_src()}/temp/"):
        if str(img).startswith("price"):
            total_part = detection_model.extract_text(
                img=img
                )
            total += total_part

    return round(total, 2)


def write_invoice(paypal_email: str, hourly_rate: float, hours_worked: float, cost_deduction: float) -> None:
    """ 
    Writes the whole invoice

    Parameters
    ----------
    paypal_email : str
        Email which the user left for future payment
    hourly_rate : float
        Hourly rate which the employee charges
    hours_worked : float
        Hours that the employee worked
    cost_deduction : float
        Deduction of single costs, e.g. Oil was bought but
        only 5 from 10 liters are used -> The price from the
        other 5 liters is substracted from the total

    Returns
    -------
    None
    """
    st.session_state.invoice.pipeline(
        email=paypal_email,
        part_cost=get_total_price(),
        hourly_rate=hourly_rate,
        hours_worked=hours_worked,
        cost_deduction=cost_deduction
    )

    return None


# def ping_streamlit(url: str ="https://rechnung.streamlit.app/Logs", ping_interval: int =86_400) -> NoReturn:
#     """ 
#     Pings streamlit app to keep it active

#     Parameters
#     ----------
#     url : str, optional (default="https://rechnung.streamlit.app/")
#         URL of app which should be pinged
#     ping_interval : int, optional (default=86_400)
#         Interval in which the app should be pinged (1 day)
    
#     Returns
#     -------
#     NoReturn
#     """
#     while True:
#         try:
#             #Send a GET request to streamlit app
#             response = requests.get(url=url)

#             if response.status_code == 200:
#                 log.get_logger().info(f"Pinged {url} successfully")
#             else:
#                 log.get_logger().error(f"Failed to ping {url}")
            
#         except Exception as e:
#             log.get_logger().error(f"Error pinging {url}: {e}")

#         time.sleep(ping_interval)


detection_model = load_detection_model()
# -------Streamlit------- #
def main() -> None:
    """ 
    Main streamlit visualization

    Returns
    -------
    None
    """
    finished = False
    st.header('Rechnung erstellen', divider="grey")

    paypal_email = st.text_input(
        "PayPal E-Mail"
    )
    hourly_rate = st.number_input(
        "Stundenlohn (Euro)", min_value=10.0, step=0.5, format="%.2f"
    )
    hours_worked = st.number_input(
        "Gearbeitete Zeit (Stunden)", min_value=0.0, step=0.5, format="%.2f"
    )
    cost_deduction = st.number_input(
        "Abzug Einzelkosten", min_value=0.0, step=0.5, format="%.2f"
    )

    with st.form("my-form", clear_on_submit=True):
        uploaded_images = st.file_uploader(
            "Rechnungen auswählen", type=["jpg", "jpeg", "png"], accept_multiple_files=True
            )

        submitted = st.form_submit_button("Upload files!")

        if len(uploaded_images) > 0 and submitted:
            with st.spinner("Wait for it ..."):
                get_roi(uploaded_images=uploaded_images)
                write_invoice(
                    paypal_email=paypal_email,
                    hourly_rate=hourly_rate,
                    hours_worked=hours_worked,
                    cost_deduction=cost_deduction
                    )
            st.success("Done!")
            finished = True

    if finished is True:
        with open("invoice_final.pdf", 'rb') as f:
            st.download_button("Download Invoice", f, file_name="invoice_final.pdf")
        finished = False



if __name__ == '__main__':
    main()
    # ping_streamlit()
