# **Video Deepfake Detection Master's Graduation Project**
Undertook a comprehensive exploration of fake and real video datasets, employing advanced techniques in face detection, data preprocessing, and the creation of structured training, validation, and testing sets.This project holds significance as it served as the culmination of my Master's degree in Ottawa in 2023.

- [Google Colab Pro+](https://colab.google/): Ensure you have access to Colab Pro+ for enhanced features.
- Required libraries: scikit-learn, pandas, matplotlib.
- Execute cells in a Jupyter Notebook environment.
- Processing power needed (GPU).
- Libraries:
   + OpenCV: Version installed: 4.8.1.78        + NumPy: Version 1.24.3
   + cvlib	                                    + Matplotlib: Version 3.7.2
   + TensorFlow: Version: 2.15.0                + Kears: Version 2.15.0
   + scikit-learn: Version: 1.3.0	


## **Design Overview**
The deepfake detection system utilizes a multi-stage approach involving data preprocessing, feature extraction, deep learning-based classification, and a user-friendly web interface. It employs state-of-the-art algorithms to distinguish between authentic and manipulated videos, addressing the challenge of deepfake proliferation.
 <p align="center">
   <img src="https://github.com/RimTouny/Video-Deepfake-Detection-Masters-Graduation-Project/assets/48333870/c1a6a0f7-4e95-4e55-bc09-3b43d6e4426b">
 </p>
 

### Dataset Description:
 - **Dataset Overview:**
   - Compilation of 2000 videos.
   - Each video duration ranges between 8 to 13 seconds.
 - **Data Sources:**
   - Combination of sponsor-contributed data (under confidentiality agreements).
   - Internally generated data using the ROOP Face Swap technique.
 - **Dataset Composition:**
   - Balanced composition with 1000 authentic videos.
   - Includes 1000 deep fake simulations.
 - **Realism and Applicability:**
   - Diverse subjects featured in the videos.
   - Encompasses both celebrities and ordinary individuals.
   - Enhances realism and ensures broader applicability of the dataset.
 <p align="center">
   <img src="https://github.com/RimTouny/Video-Deepfake-Detection-Masters-Graduation-Project/assets/48333870/8f763783-3603-478d-9430-2bbf656ef734">
 </p>
 
`Image` 
## **Key Tasks Undertaken**    
  1. **Data Loading:**
