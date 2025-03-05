# CamScanner
CamScanner is a Python-based application that detects the edges of a document from an image, applies perspective correction, and extracts a clean scanned version. This project is built using OpenCV and NumPy and does not rely on any external APIs.

Features
Automatic Edge Detection – Identifies the document’s boundaries in an image
Perspective Transformation – Warps and aligns the document to a proper rectangular shape
Noise Reduction – Applies image processing techniques to enhance the scanned output
Lightweight and Fast – Runs efficiently on local machines without external dependencies

Steps to Use

Download the code from CamScanner repository 
Extract the ZIP file
Ensure to install the required dependencies:
pip install opencv-python numpy imutils
Run the Code
Open a terminal or command prompt
Navigate to the project directory
Run the script:
python document_scanner.py
Enter the image path when prompted
The processed document will be displayed and saved as scanned_document.jpg
