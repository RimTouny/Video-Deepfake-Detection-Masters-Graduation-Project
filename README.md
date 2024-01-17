# **Video Deepfake Detection Master's Graduation Project**
Undertook a comprehensive exploration of fake and real video datasets, employing advanced techniques in face detection, data preprocessing, and the creation of structured training, validation, and testing sets.This project holds significance as it served as the culmination of my Master's degree in Ottawa in 2023.

- [Google Colab Pro+](https://colab.google/): Ensure you have access to Colab Pro+ for enhanced features.
- Required libraries: scikit-learn, pandas, matplotlib.
- Execute cells in a Jupyter Notebook environment.
- Processing power needed (GPU).
- Libraries:
   + OpenCV: Version installed: 4.8.1.78
   + NumPy: Version 1.24.3
   + cvlib
   + Matplotlib: Version 3.7.2
   + TensorFlow: Version: 2.15.0
   + Kears: Version 2.15.0
   + scikit-learn: Version: 1.3.0	

### **Binary classification:** Detect and classify Deepfake videos: Real or Fake.

### **Design Overview**
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
 
## **Key Tasks Undertaken**    
   1. **Data Collection**
      - Mixture of sponsor-contributed data data and ROOP Face Swap technique-generated data.
     
   2. **Data Exploration**
      - Assess real and fake video counts.
        ```python
            Number of Fake Videos: 1000
            Number of Real Videos: 1000
         ```
   3. **Video Processing**
      - Extract frames at 1 Frame per 1s, 2s, 4s intervals.
        + 1 Frame per 1s
          ```python
            Capture one frame every 1 seconds
            Total number of videos: 1999
            Total number of frames: 16370
            Average frames per video: 8.189094547273637
          ```
         + 1 Frame per 2s
           ```python
               Capture one frame every 2 seconds
               Total number of videos: 1999
               Total number of frames: 7965
               Average frames per video: 3.9844922461230614
           ```
         + 1 Frame per 4s
           ```python
               Capture one frame every 4 seconds
               Total number of videos: 1999
               Total number of frames: 3258
               Average frames per video: 1.629814907453727
            ```
      - Resize frames to 128x128 pixels.
      - Store metadata: `Video ID`, `Frame ID`, `Video Label`.
      - Face detection using `cvlib`, resizing images to 300x300, and drawing rectangles around faces.
   
   4. **Data Preprocessing**
      - Normalize pixel values to [0, 1].
      - Label encoding for fake/real using `LabelEncoder`.
      - Split data into 80% training, 10% validation, 10% testing (1 Frame per 1s, 2s, 4s intervals).
   
   5. **Data Preparation**
      - Convert `Normalized Frame` data and `Labels` columns to TensorFlow tensors.

   6. **Model Creation and Training**
      - ResNet50, InceptionResNetV2, MobileNetV2, VGG16 models pre-trained on ImageNet.
      - Transfer learning with specific architectures(custom Layers)
        ```python
          x = GlobalAveragePooling2D()(resnet_model.output)
          x = Dense(512, activation='relu')(x)
          x = Dropout(0.5)(x) 
          x = Dense(2, activation='softmax')(x)
        ```
      - Model compilation:
        + ResNet50 Model
           ```python
             custom_optimizer = Adam(learning_rate=0.0001)   
             model.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
           ```
        + InceptionResNetV2 Model
           ```python
             lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
             optimizer = Adam(learning_rate=lr_schedule)
             model.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
           ```
        + MobileNetV2 Model
          ```python
             sgd = SGD(lr=0.0001)  # Stochastic Gradient Descent optimizer with a specific learning rate
             vgg_model_transfer.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])  # Compile the model

          ```
      - Training details: epochs, batch size, early stopping.
        ```python
           epochs=100
           batch_size=32
           learning rate= 0.00001
           early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
        ```

   
   7. **Evaluation and Result Analysis**
      - Confusion matrix for video label determination: Calculated based on a specific threshold for determining the video label (REAL or FAKE) from the predicted frames.
          <p align="center">
            <img src="https://github.com/RimTouny/Video-Deepfake-Detection-Masters-Graduation-Project/assets/48333870/3da2ced4-736e-4ed0-a223-6e7b42579026">
          </p>
         
         + Prediction Threshold:
            - The evaluation process primarily relies on counting the occurrence of the Real Label among the predicted frames for each video.
            - If over 70% of the frames are predicted as REAL, the video is categorized as REAL; otherwise, it is classified as FAKE.

         + Categorization of Videos:
            - For each video, the model predicts a label based on the majority class of the frames. If more than 70% of the frames are predicted as REAL, the video is categorized as REAL; otherwise, it is categorized as FAKE.

         + Comparison with Actual Labels:
            - Subsequently, the predicted label obtained from the majority prediction of frames for each video is compared with the actual video label.
       <p align="center">
         <img src="https://github.com/RimTouny/Video-Deepfake-Detection-Masters-Graduation-Project/assets/48333870/0ae81b1d-a233-4379-8519-f8e4825595e3">
       </p>
           Here is an example illustrating our evaluation process on the ResNetV2 model using the Test Set in one frame per 1 sec. The green column (Actual Label) contains the known actual labels of each video, while the red column (Model Decision) is derived from the two blue columns (Predicted Fake Count, Predicted Real Count).

      - F1 score calculation based on video-level predictions and labels.
      - Learning and loss curve analysis  based on video-level predictions and labels.
      - Model comparison to identify champion model.
           <p align="center">
            <img src="https://github.com/RimTouny/Video-Deepfake-Detection-Masters-Graduation-Project/assets/48333870/951619d5-30fc-4835-8c14-87633d0aa90b">
          </p>
         + 1-Second Superiority
            Selecting a One frame per 1-second duration for video processing is recommended due to its consistent high
            training and validation accuracy across different models (ResNetV2, InceptionResNetV2, MobileNetV2, VGG-16).This duration strikes a balance between capturing essential temporal information, ensuring better generalization, and
            reducing computational load for improved efficiency in training and inference.
              ```python
               <function keras.src.applications.mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000, classifier_activation='softmax', **kwargs)>
              ```

   8. **Cross-Validation**
      - Model selection based on Model Comparison results: MobileNetV2 was selected for cross-validation.
      - 5-fold split maintaining class distribution.
      - For each fold, the model is trained on a subset of the data and evaluated on the validation set and calculate accuracy, F1 score and trained models and training history for each fold
         <p align="center">
            <img src="https://github.com/RimTouny/Video-Deepfake-Detection-Masters-Graduation-Project/assets/48333870/71ca8398-2e60-416e-bb96-c262c651dfdc">
          </p>

   9. **Soft Voting**
      - Predict probabilities for top 3 models:ResNet50, InceptionResNetV2 and MobileNetV2 models.
      - Apply threshold of 0.5 for binary predictions.
         <p align="center">
            <img src="https://github.com/RimTouny/Video-Deepfake-Detection-Masters-Graduation-Project/assets/48333870/118e6329-9c95-43bf-890d-84a72bd2c17b">
          </p>

   10. **Hyperparameters Tuning**: on Chapion Model MobileNetV2 Model 
       - Different Learning Rates: with batch size 32 and early stop after 5 epochs.
            <p align="center">
               <img src="https://github.com/RimTouny/Video-Deepfake-Detection-Masters-Graduation-Project/assets/48333870/6f26dde0-8600-4034-80df-bb53b1099c2b">
            </p>  

       - Different Batch Sizes: with Learning Rate 10^(-4) and early stop after 5 epochs.
             <p align="center">
               <img src="https://github.com/RimTouny/Video-Deepfake-Detection-Masters-Graduation-Project/assets/48333870/627c17d6-5eb4-4842-b658-bfcf88899c2f">
            </p>    

       - Different number of epochs in early stop: with Learning Rate 10^(-4) and batch size =32.
          <p align="center">
            <img src="https://github.com/RimTouny/Video-Deepfake-Detection-Masters-Graduation-Project/assets/48333870/8f6a282f-a432-4fdb-82df-e07eb6689509">
         </p>    

   12. **Overall Comparison and The Superior Model**
      - Save the superior model for further development.
          <p align="center">
            <img src="https://github.com/RimTouny/Video-Deepfake-Detection-Masters-Graduation-Project/assets/48333870/1975e945-0101-4390-9c6f-989b7d21225e">
         </p> 
          <p align="center">
            <img src="https://github.com/RimTouny/Video-Deepfake-Detection-Masters-Graduation-Project/assets/48333870/f10e410c-96b9-4478-b7d3-dbbedf4f4602">
         </p> 
   
   14. **Deployment Phase**
      - Implement deployment using Flask.
