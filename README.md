# Digital Image Processing - Python Projects  
**Institution**: Faculdade de Ciências da Universidade de Lisboa  
**Course**: Digital Image Processing  
**Project Date**: 2021  
**Project Language**: Portuguese

## Project Overview
This repository contains two Python-based image processing projects that demonstrate various techniques for noise reduction, image filtering, and feature extraction using spatial and Fourier domains. The projects focus on real-world applications like removing noise from images and detecting specific features such as riverbeds, roads, and roofs in satellite images.

## Skills and Tools Used
- **Software**: Python, NumPy, SciPy, Matplotlib, scikit-image (`skimage`)
- **Techniques**: Spatial filtering, Fourier filtering, histogram thresholding, watershed transformation, morphological operations
- **Applications**: Noise reduction, riverbed detection, road detection, roof detection

---

## Project 1: Noise Attenuation/Elimination

### Description
In this project, various spatial and Fourier filters are applied to remove noise from the image `noisyland.tif`. The goal is to perform noise attenuation/elimination and visually display the results.

### Key Techniques
- Noise attenuation using spatial filters (e.g., Gaussian, Median).
- Noise attenuation using Fourier filters (e.g., Low-pass, High-pass).
- Displaying the original noisy image, the applied filter, and the filtered image.

### Files
- `/project1/`: Folder containing the following files:
  - `noise_reduction.py`: Python script for applying noise reduction techniques.
  - `noisyland.tif`: Input noisy image.
  - `filtered_output.png`: Output image after applying filters.

---

## Project 2: Image Segmentation and Feature Extraction

### Description
This project focuses on image segmentation and feature extraction from satellite images, involving several key tasks such as riverbed extraction, watershed transformation, and road/roof detection.

### Key Techniques
- Binary extraction of riverbed areas.
- Watershed transformation for riverbed central line detection.
- Histogram thresholding for segmentation and extraction of roads and roofs.

### Files
- `/project2/`: Folder containing the following files:
  - `segmentation.py`: Python script for image segmentation tasks.
  - `lsat01.tif`: Satellite image for riverbed extraction.
  - `ik02.tif`: RGB satellite image for road and roof detection.
  - `/output_images/`: Contains images showing extracted features.

---

## Dependencies
Both projects rely on the following Python libraries:
- `numpy`
- `scipy`
- `matplotlib`
- `scikit-image` (`skimage`)
- `imageio`

To install the dependencies, you can run:
```bash
pip install -r requirements.txt
```

---


## Key Results

- **Project 1:** 
  - Successful noise attenuation in both spatial and Fourier domains.
  - Visual comparison of the original noisy image and the filtered image using different techniques.
  
- **Project 2:**
  - Extraction of riverbeds using binary segmentation and watershed transformation to obtain the river's centerline.
  - Successful detection of roads and roofs in satellite imagery using histogram thresholding and morphological operations.
