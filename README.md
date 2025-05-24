# LeafSense
Leafsense : AI POWERED SYSTEM FOR PLANT DISEASE DETECTION AND PERSONALIZED CARE  

# Plant Disease Diagnosis and Care Assistant

A dark-themed plant disease diagnosis and care management website with image analysis and PDF reporting capabilities. This application helps farmers and plant owners diagnose plant diseases from images and generate personalized care routines.

## Features

- **Disease Diagnosis**: Upload a plant image and get an AI-powered diagnosis with treatment recommendations
- **Plant Care Routine**: Generate personalized care schedules based on plant type and age
- **PDF Reports**: Download detailed PDF reports for both diagnosis and care recommendations
- **Dark Theme**: Easy on the eyes with a professional, sleek design

## Technical Implementation

The application uses:
- Flask for the web framework
- SQLAlchemy for database operations
- EfficientNetB0 for image classification (trained on plant disease dataset)
- ReportLab for PDF generation
- Dark-themed custom CSS for the UI

## Training the Plant Disease Model

To train the model with the PlantVillage dataset from Kaggle:

1. **Install Required Dependencies**:
   ```
   pip install tensorflow scikit-learn kaggle pillow
   ```

2. **Set Up Kaggle API**:
   - Go to your Kaggle account settings
   - Create a new API token and download kaggle.json
   - Place this file in `~/.kaggle/` or use the `KAGGLE_CONFIG_DIR` environment variable

3. **Download the Dataset**:
   ```
   python download_dataset.py
   ```
   
   Alternatively, you can manually download the dataset from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) and extract it:
   ```
   python download_dataset.py --manual-path /path/to/downloaded/zip --extract-to data
   ```

4. **Prepare and Train the Model**:
   ```
   python train_model.py
   ```
   
   This will:
   - Prepare the dataset (split into train/val/test sets)
   - Create and train the EfficientNetB0 model
   - Save the trained model to `models/plant_disease_efficientnetb0.h5`
   - Save the class indices to `models/plant_disease_efficientnetb0_classes.npy`

5. **Use the Model in the Application**:
   - The application will automatically use the trained model if it's available
   - If the model isn't available, it falls back to a simulated prediction mode

## Model Architecture

The model uses EfficientNetB0 as a base, with:
- Pre-trained ImageNet weights
- A two-phase training approach (feature extraction, then fine-tuning)
- Custom classification head with 38 classes (PlantVillage dataset)
- Data augmentation to improve model robustness

## Dataset

The model is designed to be trained on the Plant Disease dataset from Kaggle:
[Plant Disease dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

This dataset contains 38 different classes of plant diseases, with over 20,000 images.



![Screenshot 2025-05-18 152139](https://github.com/user-attachments/assets/5fbe6804-9a94-4875-b009-0de2cbf28ac5)



![Screenshot 2025-05-18 151927](https://github.com/user-attachments/assets/dbafc40d-e043-4270-b494-39b32258b4ac)

