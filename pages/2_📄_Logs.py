""" 
File containing all functions for showing the logs
from logs.log in streamlit

File is written in pylint standard
"""

import streamlit as st



def main(log_path: str) -> None:
    """
    Visualizes logs from log-file

    Parameters
    ----------
    log_path : str
        Path to the log-file

    Returns
    -------
    None
    """

    st.title('Log File Monitor')
    log_container = st.container(border=True, height=500)
    
    with open(log_path, "r") as file:
        for line in file:
            if "invoices" in line:
                log_container.write("\n")
            log_container.write(line)

    return None



if __name__ == "__main__":
    st.set_page_config(page_title="Delivery Note Reader", layout="wide")
    log_file_path = "logs.log"
    main(log_path=log_file_path)
