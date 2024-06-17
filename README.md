# Deep.Imager

![Screenshot 2024-06-17 131144](https://github.com/CodeStrate/ImageUpscaler/assets/56911705/35c08efe-64cd-4a91-9cb5-7b76d4b152ba)

Deep.Imager is a Python application powered by Streamlit that upscales images without any loss in quality. This app implements traditional upscaling methods and incorporates advanced deep learning models to provide high-quality image enhancement.

## Features

- **Traditional Upscaling Methods**: (Including my own albeit slower implementations)
  - **Bicubic Interpolation** 
  - **Lanczos Interpolation**

- **Deep Learning Models**:
  1. **EDSR_x4.pb**: Enhanced Deep Residual Networks for Single Image Super-Resolution (increases resolution by 4x).
  2. **ESPCN_x4.pb**: Efficient Sub-Pixel Convolutional Neural Network for Real-Time Super-Resolution (increases resolution by 4x).
  3. **FSRCNN_x3.pb**: Fast Super-Resolution Convolutional Neural Network (increases resolution by 3x).
  4. **LapSRN_x8.pb**: Deep Laplacian Pyramid Networks for Fast and Accurate Image Super-Resolution (increases resolution by 8x).
 

  ![image](https://github.com/CodeStrate/ImageUpscaler/assets/56911705/c62237b7-0d5d-4853-973d-041f59a578bd)

  ![image](https://github.com/CodeStrate/ImageUpscaler/assets/56911705/490ff274-4b6e-4b10-85a9-df5b22836510)



- **Additional Features**:
  - Gaussian sharpening kernel for enhanced image details.
  - Download the output image as a PNG file.

## Installation

To run Deep.Imager, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ImageUpscaler.git
   cd ImageUpscaler
   ```
2. Create a Python Virtual Environment
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```
3. Install requirements.txt
   ```bash
     pip install -r requirements.txt
   ```
4. Run app.py
   ```bash
     streamlit run app.py
   ```
## Usage
1. Upload an image using the file uploader.
2. Choose the desired upscaling method (traditional or deep learning model).
3. Optionally, apply the Gaussian sharpening filter.
4. View the upscaled image and download it as a PNG file.


## Contributing
I welcome contributions! Please feel free to submit issues, feature requests, or pull requests.


## Acknowledgements
- **EDSR**: Enhanced Deep Residual Networks for Single Image Super-Resolution
- **ESPCN**: Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
- **FSRCNN**: Accelerating the Super-Resolution Convolutional Neural Network
- **LapSRN**: Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks


