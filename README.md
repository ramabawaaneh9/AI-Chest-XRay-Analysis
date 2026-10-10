 AI Chest X-Ray Analysis

Lung Segmentation, Multi-Class Classification & Explainable Deep Learning

An end-to-end Deep Learning project for chest X-ray analysis, combining U-Net lung segmentation, automated lung-region analysis, ResNet-based multi-class classification, and Grad-CAM explainability.

The project explores how segmentation and classification models can be evaluated alongside image-based quantitative features and visual explanations, using chest X-ray images from four diagnostic categories.

Disclaimer: This is an educational and research project. It is not intended for clinical diagnosis, medical decision-making, or replacement of professional medical evaluation.

Table of Contents

Project Overview

Project Pipeline

Dataset

. Dataset Preparation and Preprocessing

. U-Net Lung Segmentation

. U-Net Visualization
    
. U-NET MASK-BASED FEATURE EXTRACTION

. Automated Lung Segmentation Analysis

. Chest X-Ray Classification

. Classification Results

. ACM Analysis

. Grad-CAM Explainability

. Quantitative Grad-CAM Evaluation

Technologies Used

Repository Structure

Limitations and Future Improvements

Authors

 Project Overview

Chest X-ray images are widely used in medical imaging research. Deep Learning can support image segmentation and classification, while explainability techniques can help researchers inspect which image regions influence model predictions.

This project implements a multi-stage workflow:

Lung Segmentation: A U-Net model predicts binary lung masks from chest X-ray images.

U-Net Mask-Based Feature Extraction — Extract quantitative features from the predicted masks and corresponding X-ray images.

Automated Lung Analysis: Quantitative features are extracted from predicted masks and the corresponding grayscale images.

Multi-Class Classification: ResNet18 and ResNet34 classify X-ray images into four categories.

Explainable AI: Grad-CAM produces visual explanations of classification predictions.

Quantitative Evaluation: Classification performance and attention-related measurements are analyzed to compare model behavior.

The goal is to explore both predictive performance and the spatial behavior of model attention, rather than reporting classification accuracy alone.

 Project Pipeline:
 
             Dataset Preparation
                     |
                     ↓
             Data Preprocessing
                     |
                     ↓
          U-Net Lung Segmentation
                     |
                     ↓
             U-Net Visualization
                     |
                     ↓
     U-Net Segmentation Results Analysis
                     |
                     ↓
      U-Net Mask-Based Feature Extraction
                     |
                     ↓
  AUTOMATED LUNG SEGMENTATION ANALYSIS
                     |
                     ↓
       ResNet18 & ResNet34 Training
                     |
                     ↓
          Multi-Class Classification
                     |
                     ↓
                ACM Analysis
                     |
                     ↓
            Grad-CAM Visualization
                     |
                     ↓
       Quantitative Grad-CAM Evaluation

Suggested visual: A diagram showing the complete project workflow: Dataset Preparation → Data Preprocessing → U-Net Lung Segmentation → U-Net Visualization → U-Net Segmentation Results Analysis → U-Net Mask-Based Feature Extraction → Automated Lung Segmentation Analysis → ResNet18 & ResNet34 Training → Multi-Class Classification → ACM Analysis → Grad-CAM Visualization → Quantitative Grad-CAM Evaluation.
 
 The diagram should reflect the actual sequence of stages implemented in the notebook.
 
 Dataset:
The project uses the COVID-19 Radiography Database, containing chest X-ray images organized into four categories:

COVID

Lung_Opacity

Normal

Viral Pneumonia

The original dataset contained the following number of images:

Class

Original Images

Lung_Opacity

6,012

Normal

10,192

Viral Pneumonia

1,345

COVID

3,616

Total

21,165

The dataset also provides lung masks used for segmentation training and evaluation.

Dataset source:[COVID-19 Radiography Database — Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

Please follow the dataset provider's terms of use and citation requirements.

1. Dataset Preparation and Preprocessing

The preprocessing stage prepares the images for the subsequent segmentation and classification experiments.

Selective Downsampling

To reduce the imbalance between the dataset categories, random downsampling was applied to the two largest classes.

The implemented preprocessing configuration uses:

Random seed: 42

Maximum images per selected class: 3,000

Downsampled classes: Normal and Lung_Opacity

COVID and Viral Pneumonia: Retained without downsampling in this stage

The expected class distribution after this operation is:

Class

Images After Downsampling

Lung_Opacity

3,000

Normal

3,000

Viral Pneumonia

1,345

COVID

3,616

Total

10,961

The final distribution is not fully balanced because the downsampling rule applies only to Normal and Lung_Opacity.

<img width="405" height="203" alt="image" src="https://github.com/user-attachments/assets/57cbe272-985e-4c15-a450-027369deff7e" />


2. U-Net Lung Segmentation

Objective:

The U-Net model is trained to predict binary lung masks from chest X-ray images. These masks identify the image regions classified as lung tissue by the segmentation model.

Methodology

The segmentation workflow includes:

Loading chest X-ray images and their corresponding masks.

Resizing the images and masks to a consistent spatial resolution.

Splitting the dataset into training 70%, validation 15%, and test subsets 15%.

Training the U-Net model using PyTorch.

Selecting the best model checkpoint based on validation Dice performance.

Evaluating the selected model on the test set.

Training Configuration

Parameter

Configuration

Model

U-Net

Framework

PyTorch

Input resolution

256 × 256

Batch size

32

Number of epochs

10

Optimizer

Adam

Learning rate

0.001

Output activation

Sigmoid

Binary mask threshold

0.5

Checkpoint selection

Best validation Dice

The model predicts a probability map, which is thresholded at 0.5 to obtain a binary segmentation mask.

Evaluation Metrics

Dice Coefficient

Measures the overlap between the predicted segmentation and the ground-truth mask.

Intersection over Union (IoU)

Measures the ratio of the intersection to the union of the predicted and ground-truth regions.

Pixel Accuracy

Measures the proportion of pixels classified correctly. It should be interpreted alongside Dice and IoU because background pixels can dominate segmentation images.

Segmentation Results

Metric

Reported Result

Test Dice

94.70%

Test IoU

90.41%

Test Accuracy

97.72%

3. U-Net Visualization

Quantitative metrics are complemented by visual inspection of the predicted segmentation masks.

The visualizations compare the original chest X-ray images with their corresponding ground-truth masks and U-Net predictions.

Visual Comparison

 Sample 146 | Dice: 0.9912 | IoU: 0.9826

<img width="1662" height="508" alt="Project Pipeline" src="https://github.com/user-attachments/assets/2970c865-0b22-4f85-a24e-f14e400a7372" />

Sample 1273 | Dice: 0.9884 | IoU: 0.9770

<img width="1890" height="562" alt="image" src="https://github.com/user-attachments/assets/a764f111-22cd-4581-9ea7-2339d82af34c" />

Recommended figure layout:

Original X-ray

Ground-truth lung mask

Predicted lung mask

Overlay or contour comparison

These comparisons help illustrate the segmentation model's behavior, including cases where predicted boundaries align with the reference masks and cases where errors are visible.

4. Extracted Features

Feature

Description

lung_area_ratio

Proportion of image pixels classified as lung tissue.

num_lung_regions

Number of connected components detected in the binary mask.

mean_lung_intensity

Mean grayscale intensity of pixels inside the predicted lung mask.

largest_lung_region_ratio

Proportion of the image occupied by the largest connected component.

Feature Extraction Results

The test-set feature dataset contains 1,645 images and four extracted features, plus an image identifier.

Feature

Mean

Standard Deviation

Lung Area Ratio

0.2306

0.0632

Number of Lung Regions

11.19

9.40

Mean Lung Intensity

0.4215

0.1010

Largest Lung Region Ratio

0.1255

0.0340

Data Quality

Total feature records: 1,645

Feature columns: 4

Identifier column: 1

Missing values: 0 across all columns

Note: These features describe image regions and segmentation outputs. They are exploratory measurements, not validated clinical biomarkers.

5. Automated Lung Segmentation Analysis

This stage combines U-Net lung segmentation with quantitative feature extraction to analyze predicted lung regions in chest X-ray images.

 Evaluation and Visualization:

The system evaluates predicted masks against ground-truth masks using **Dice Coefficient** and **Intersection over Union (IoU)**.

For each selected sample, the generated report displays the original X-ray, ground-truth mask, predicted mask, segmentation overlay, extracted lung region, and quantitative feature values.

The analysis processes four randomly selected test samples using a fixed random seed for reproducibility.

<img width="1893" height="528" alt="image" src="https://github.com/user-attachments/assets/e49a1506-737a-48a9-891a-2515809243d0" />

6. Chest X-Ray Classification

Objective:

The classification stage evaluates two convolutional neural network architectures for categorizing chest X-ray images into four classes.

Models

ResNet18

A residual convolutional neural network with 18 layers, evaluated as one of the classification models.

ResNet34

A deeper residual network with 34 layers, evaluated to compare classification performance and Grad-CAM behavior against ResNet18.

Both models are evaluated on the same four-category classification task.

Classification Categories for ResNet18:

Class Label:

COVID

Lung_Opacity

Normal

Viral Pneumonia

Classification Categories for ResNet34:
 
 Class Label:
 
COVID

Lung_Opacity

Normal

Viral Pneumonia

Classification Workflow:

Prepare the classification dataset and labels.

Split the data into training 70%, validation 15%, and test sets 15%.

Train and evaluate ResNet18 and ResNet34.

Select the best checkpoint using validation performance.

Compare classification metrics on the test set.

Generate Grad-CAM explanations for eligible samples.

Model Comparison

7. Classification Results

The following results summarize the reported test-set performance of the two models.

Overall Performance

Metric

ResNet18: 

Test Accuracy 

92.71%

Test Weighted F1-score

92.73% 

Best Validation F1-score

92.47%

ResNet34:

Test Accuracy

93.07%

Test Weighted F1-score

93.06%

Best Validation F1-score

92.00%

ResNet34 achieved slightly higher reported test accuracy and weighted F1-score, while ResNet18 achieved the higher validation F1-score.

This comparison illustrates why model selection should consider validation performance, test performance, class-wise results, and the intended application rather than a single metric.

Test Accuracy by Class

The class-wise figures below are calculated from the samples reported in the classification evaluation.

Class

ResNet18 Accuracy: 

COVID

94.32%

Lung_Opacity

90.36%

Normal

93.59%

Viral Pneumonia

93.07%

ResNet34 Accuracy:

COVID

91.81%

Lung_Opacity

91.28%

Normal

89.29%

Viral Pneumonia

98.02%

The reported ResNet18 class counts sum to 1,613 eligible Grad-CAM samples, while the ResNet34 class counts sum to 1,353 eligible Grad-CAM samples. These class-wise figures should therefore be interpreted in the context of the corresponding evaluation outputs rather than assumed to represent identical subsets.

Confusion matrix for ResNet18:

<img width="923" height="726" alt="image" src="https://github.com/user-attachments/assets/75cb18fb-a94b-4038-a008-762f86e4ded7" />

Confusion matrix for ResNet34:

<img width="921" height="729" alt="image" src="https://github.com/user-attachments/assets/c40ebd1a-7b0e-4e17-a1a4-ad12e0fe3cd2" />

 8. Quantitative Grad-CAM Analysis

This stage evaluates how ResNet18 and ResNet34 distribute their attention inside and outside the ground-truth lung regions using Grad-CAM.

| Metric | ResNet18 (Test) | ResNet34 (Test) |
| Classification Accuracy | 92.71% | 93.07% |
| Grad-CAM Coverage | 98.05% | 82.25% |
| Mean Attention Inside Lungs | 46.71% | 36.24% |
| Mean Attention Enrichment | 2.09× | 1.60×|

Key Findings:

- Classification: ResNet34 achieved the higher test accuracy of 93.07%, compared with 92.71% for ResNet18.
- Lung-focused attention: ResNet18 showed stronger attention enrichment within lung regions, suggesting more concentrated attention relative to lung area.
- Grad-CAM reliability: ResNet18 produced valid Grad-CAM results for a larger proportion of test samples.
- Attention limitations: Both models assigned substantial attention outside the lung regions, highlighting the importance of evaluating potential reliance on non-lung features.
 
Note: Attention enrichment is an exploratory metric, not proof that a model focuses on medically relevant features. These results evaluate model behavior and do not establish clinical reliability.

8.1 Grad-CAM Explainability

Objective

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to generate visual explanations of classification predictions.

It highlights image regions that contribute to a selected class score, helping researchers inspect how a trained model responds to the input image.

Grad-CAM is applied to the classification models to compare their attention patterns.

Visual Analysis

<img width="970" height="598" alt="image" src="https://github.com/user-attachments/assets/6e72db79-4bea-44b8-82cd-a8a375eb54d0" />
<img width="856" height="491" alt="image" src="https://github.com/user-attachments/assets/0485e309-a685-4c35-8b2f-3947446ccb87" />
<img width="855" height="486" alt="image" src="https://github.com/user-attachments/assets/33fabe11-c3a7-4e3d-8b44-f790bd347f9f" />
<img width="980" height="593" alt="image" src="https://github.com/user-attachments/assets/86b7c1c1-ac57-4d93-b5a2-0ad4e6720e20" />

 8.2 Grad-CAM Visual Comparison

This final stage provides visual explanations of ResNet18 and ResNet34 predictions on the same chest X-ray images, including correctly classified and misclassified examples.

 Visualizations

- Original X-ray: Displays the original chest X-ray image.
- U-Net Lung Segmentation: Compares the predicted lung mask with the ground-truth mask using Dice and IoU scores.
- ResNet18 Grad-CAM: Highlights image regions influencing the ResNet18 prediction.
- ResNet34 Grad-CAM: Highlights image regions influencing the ResNet34 prediction.
- Side-by-Side Comparison: Displays both models' attention maps on the same image, alongside their predicted classes and confidence scores.

 Example Analysis

The examples include correct and incorrect predictions selected using ResNet18 and ResNet34 . Both classifiers are then evaluated on the same images, allowing a direct visual comparison of their predictions and attention patterns.

correct example:

<img width="1892" height="1076" alt="image" src="https://github.com/user-attachments/assets/358db57f-cc67-42f9-b259-257bfb3d84e7" />


incorrect example:

<img width="1888" height="1079" alt="image" src="https://github.com/user-attachments/assets/385046c1-bc24-4251-8074-a610c86cd5c4" />


Important Note: Grad-CAM visualizations help interpret model behavior but do not prove that the highlighted regions represent medically meaningful features. These visualizations are intended for research and educational purposes, not clinical decision-making.

8.3 Quantitative Grad-CAM Evaluation

In addition to visual inspection, the project evaluates Grad-CAM attention patterns relative to the available lung masks.

This analysis examines whether attention is concentrated within the segmented lung regions and compares the results across models.

Test-Set Results

Metric

ResNet18:

Total test images

1,645

Grad-CAM-valid samples

1,613

Grad-CAM coverage

98.05%

Overall classification accuracy

92.71%

Accuracy on Grad-CAM-valid samples

92.87%

Mean attention inside lung

0.4671

Mean attention outside lung

0.5329

Mean ground-truth lung area ratio

0.2356

Mean attention enrichment

2.0855

Median attention enrichment

2.0063

Samples with enrichment greater than 1

97.64%

ResNet34:

Total test images

1,645

Grad-CAM-valid samples

1,353

Grad-CAM coverage

82.25%

Overall classification accuracy

93.07%

Accuracy on Grad-CAM-valid samples

91.80%

Mean attention inside lung

0.3624

Mean attention outside lung

0.6376

Mean ground-truth lung area ratio

0.2326

Mean attention enrichment

1.5953

Median attention enrichment

1.6350

Samples with enrichment greater than 1

86.18%

Coverage represents the percentage of test images for which the Grad-CAM evaluation produced valid samples, based on the reported counts.

Correct and Incorrect Predictions

Metric

ResNet18:

Correct predictions in Grad-CAM-valid subset

1,498

Incorrect predictions in Grad-CAM-valid subset

115

Mean enrichment — correct predictions

2.0912

Mean enrichment — incorrect predictions

2.0111

ResNet34:

Correct predictions in Grad-CAM-valid subset

1,242

Incorrect predictions in Grad-CAM-valid subset

111

Mean enrichment — correct predictions

1.5845

Mean enrichment — incorrect predictions

1.7164

These results show that the two models exhibit different attention distributions and different Grad-CAM evaluation coverage.

The mean attention values indicate that ResNet18 assigned a larger share of its measured attention inside the lung masks than ResNet34 in this experiment. However, the higher enrichment values do not by themselves establish better clinical reasoning or more trustworthy predictions.

The enrichment metric should be interpreted according to its exact implementation, including how attention maps are normalized and how lung masks are used in the calculation.

 Technologies Used

Technology

Purpose

Python

Core programming language

PyTorch

Deep Learning training and inference

U-Net

Lung segmentation

ResNet18

Chest X-ray classification

ResNet34

Chest X-ray classification

NumPy

Numerical computation

Pandas

Feature dataset construction and analysis

scikit-image

Connected-component labeling and region properties

Matplotlib

Result visualization

Seaborn

Statistical visualization

scikit-learn

Classification metrics and evaluation

Grad-CAM

Visual explanation of model predictions

Google Colab

Development and experimentation

Repository Structure

The following is a recommended structure for the published repository. Update it to match the files actually included.

AI-Chest-XRay-Analysis/
│
├── README.md
├── chest_xray_analysis.ipynb
├── requirements.txt
│
└── images/
    ├── project_pipeline.png
    ├── dataset_distribution.png
    ├── unet_metrics.png
    ├── unet_segmentation_results.png
    ├── lung_feature_analysis.png
    ├── resnet_model_comparison.png
    ├── resnet18_confusion_matrix.png
    ├── resnet34_confusion_matrix.png
    ├── acm_analysis.png
    ├── resnet18_gradcam.png
    ├── resnet34_gradcam.png
    └── gradcam_quantitative_comparison.png

The images/ folder contains selected figures used in this README. The dataset and trained model checkpoints may be kept outside the repository if they are too large or subject to distribution restrictions.

 9. Limitations and Future Improvements

Although the project combines segmentation, classification, feature extraction, and explainability, several limitations should be considered.

Current Limitations

The dataset is subject to class imbalance and sampling decisions.

Results on a single dataset do not establish generalization to other hospitals, scanners, populations, or acquisition protocols.

Segmentation quality depends on the reference masks and the quality of the predicted masks.

The extracted mask-based features have not been validated as clinical biomarkers.

Grad-CAM is a visualization technique and does not establish causal reasoning or clinical correctness.

Classification metrics alone are insufficient to establish clinical safety.

The reported Grad-CAM coverage differs between ResNet18 and ResNet34, so attention comparisons must account for the evaluated subsets.

10. Future Improvements

Evaluate segmentation and classification on external datasets.

Report per-class precision, recall, F1-score, and confusion matrices consistently.

Investigate class imbalance and calibration.

Evaluate the sensitivity of Grad-CAM findings to model architecture and input perturbations.

Compare attention against segmentation masks using clearly defined and reproducible metrics.

Add reproducible configuration files, dependency versions, and model evaluation scripts.

Develop a user interface for research demonstrations, with clear medical disclaimers.

 Authors

This project was developed collaboratively as a Deep Learning and Computer Vision project.

**Contributors: Contribute with  TOQA ABU HAMDAN , link her profile : https://github.com/toqa2024 

 Disclaimer

This repository is intended for educational and research purposes only. It is not a certified medical device, does not provide a diagnosis, and must not be used to make clinical decisions. All model predictions and visual explanations require careful interpretation and independent validation before any real-world medical application

** How to Run the Project

 1. Clone the Repository

```bash
git clone https://github.com/toqa2024/AI-Chest-XRay-Analysis.git
cd AI-Chest-XRay-Analysis
```

2. Open the Notebook

Open `chest_xray_analysis.ipynb` in Google Colab or a compatible Jupyter Notebook environment.

For Google Colab:
1. Upload or open the notebook in Colab.
2. Select a GPU runtime if available.
3. Install the required dependencies listed in `requirements.txt`.

 3. Download and Prepare the Dataset

Download the(https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) and extract it into the expected directory structure.

Update `DATASET_PATH` in the notebook to point to the dataset location.

The notebook expects the dataset to contain the four class folders, with corresponding `images` and `masks` directories.

 4. Run the Notebook

Execute the notebook cells in order, starting with dataset preparation and preprocessing.

The workflow includes:
- U-Net lung segmentation and evaluation
- Lung mask-based feature extraction
- Automated lung segmentation analysis
- ResNet18 and ResNet34 classification
- Classification evaluation and model comparison
- Grad-CAM visualization and quantitative evaluation

Some stages depend on outputs generated by earlier stages, so the notebook should be executed in the intended order.

 5. Outputs

Depending on the stages executed, the notebook generates segmentation masks, quantitative feature tables, classification metrics, confusion matrices, and Grad-CAM visualizations.

**Note:** Training time depends on the available hardware. Pretrained or previously saved model checkpoints may be required if you choose to run inference without training the models again.

Disclaimer

This project is intended for educational and research purposes only. It is not intended for clinical diagnosis or medical decision-making. 
