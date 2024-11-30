# PDF-To-Invoice
In this repository, I'm working to make your life easier when it comes to sending out final invoices for your auto repair business. Basically, I dive into these invoices and collect all the important things like tables and total prices for the parts so you don't have to worry about the little things. Just think of it as your personal invoice assistant!

# Table of content
- [Process diagramm](#process-diagramm)
- [Technology](#technology)
    - [Faster R-CNN ResNet101 architecture](#faster-r-cnn-resnet101-architecture)
    - [Pytesseract architecture](#pytesseract-architecture)
- [Setup](#setup)
    - [Virtual environment](#virtual-environment)
    - [Requirements](#requirements)
    - [Model downloading](#model-downloading)
    - [Own model](#own-model)
- [To-dos](#to-dos)
- [Troubleshooting](#troubleshooting)
- [Contact](#contact)
- [License](#license)


# Process diagramm

# Technology
We primarily rely on two key technologies: TensorFlow's Faster R-CNN ResNet101 model and the pytesseract OCR Module. Our use of TensorFlow's model is instrumental in pinpointing the total price and tables within individual invoices. These tables are then seamlessly integrated as images into the final invoice. Meanwhile, the OCR capability of pytesseract comes into play for extracting the total amount, enabling us to gather this variable and consolidate all individual values effectively.

### Faster R-CNN ResNet101 architecture
![Faster R-CNN ResNet101](./charts/faster-r-cnn_architecture.png)
<p align="center"><i>Faster R-CNN ResNet101 model architecture</i></p>

### Pytesseract architecture
![Pytesseract](./charts/ocr_flow.png)
<p align="center"><i>Pytesseract process architecture</i></p>

# Setup
### Virtual environment
To successfully execute the code in your own setup, a few prerequisites need to be met. Firstly, you'll need Python version 3.11.X to run the code smoothly and install the required packages. Additionally, for those inclined to train their own model, Python version 3.8.X is recommended. I highly advise setting up a virtual environment to keep unrelated modules at bay and maintain a focused environment for this project. This approach ensures that you can dedicate your attention solely to the relevant aspects of the task at hand.
```bash 
git clone https://github.com/Lukas-Graf/PDF-To-Invoice.git
```
```python 
python -m venv your-venv-name
```

### Requirements
The next step is to install all the requirements. Some of the most important modules are tensorflow for the models, matplotlib & opencv for visualization and flask for the web application.
```bash
pip install --upgrade pip && pip install -r requirements.txt
```

### Model downloading
Since the detection model is too large to include directly in the repository, you'll need to download it from the provided OneDrive link. Once downloaded, simply place it into the folder labeled "./res/model" within the repository directory.

### Own Model
If you want to train and setup your own model you can follow the steps in this video.<br>

[![Watch the video](https://i.ytimg.com/vi/rRwflsS67ow/sddefault.jpg)](https://www.youtube.com/watch?v=rRwflsS67ow)

# To-dos
# Troubleshooting
# Contact
# License
