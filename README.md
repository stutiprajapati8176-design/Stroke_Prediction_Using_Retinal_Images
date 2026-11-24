\# Stroke Detection Using Retinal Fundus Images by Deep Learning



\## 1. Introduction



Stroke is a serious medical condition that occurs when the blood supply to the brain is interrupted. Early identification of potential risk factors can help healthcare professionals take preventive measures.



The retina contains a network of small blood vessels that share similarities with the blood vessels found in the brain. Changes in retinal vessels, such as narrowing and abnormal patterns, may provide useful information about vascular health.



This project uses \*\*Deep Learning, Machine Learning, and Image Processing\*\* techniques to analyze retinal fundus images and identify abnormalities associated with potential stroke risk.



The system analyzes retinal blood vessels, detects vessel narrowing, identifies retinal hemorrhage, and combines these findings to generate a risk category.



\*\*Note:\*\* This project is intended for educational and research purposes. It does not provide a clinical stroke diagnosis.



\## 2. Project Objectives



\- Analyze retinal fundus images using deep learning.

\- Segment blood vessels from retinal images.

\- Extract vessel structures using skeletonization.

\- Calculate retinal blood vessel widths.

\- Detect potentially narrowed blood vessel regions.

\- Identify hemorrhage in retinal images.

\- Generate an overall risk category based on detected abnormalities.

\- Visualize the analysis results through a web application.



\## 3. Project Modules



\### 3.1 Image Acquisition



The system accepts retinal fundus images for analysis.



Publicly available retinal image datasets used in research include:



\- DRIVE

\- STARE

\- CHASE\_DB1



These datasets provide retinal images and manually labeled vessel maps that can be used as ground truth during model development.



\### 3.2 Image Preprocessing



Preprocessing improves image quality before model analysis.



Techniques include:



\- Image resizing

\- CLAHE (Contrast Limited Adaptive Histogram Equalization)

\- Green channel extraction

\- Pixel normalization

\- Retinal region masking



The green channel is used because retinal blood vessels are generally more visible in this channel.



\### 3.3 Blood Vessel Segmentation



The project uses an improved U-Net-based deep learning architecture for blood vessel segmentation.



U-Net consists of two main components:



\*\*Encoder:\*\*

\- Extracts important image features.

\- Gradually reduces spatial dimensions.

\- Captures different levels of image information.



\*\*Decoder:\*\*

\- Reconstructs the segmented output.

\- Uses extracted features to identify blood vessel regions.



Additional architectural improvements can include batch normalization, dropout, and attention mechanisms.



\*\*Output:\*\* A binary vessel segmentation image in which blood vessels are separated from the background.



\### 3.4 Feature Extraction



After segmentation, skeletonization is applied to convert blood vessels into one-pixel-wide centerlines while preserving their structure and connectivity.



The system then calculates vessel width by measuring the distance from the skeleton centerline to the vessel boundaries and doubling the distance.



These measurements help identify variations in vessel width.



\### 3.5 Blood Vessel Narrowing Detection



This module analyzes the extracted vessel width measurements.



Main steps:



1\. Vessel width analysis.

2\. Threshold-based detection.

3\. Width variation analysis.

4\. Identification of potentially narrowed regions.

5\. Visualization of detected regions in red on the original retinal image.



Vessel narrowing can be associated with vascular conditions such as hypertension and atherosclerosis.



\### 3.6 Hemorrhage Detection



This module identifies whether hemorrhage is present in the retinal image.



\*\*Model:\*\* ResUNet-based classifier.



Processing steps:



1\. Resize the retinal image to 224 × 224 pixels.

2\. Apply ImageNet normalization.

3\. Extract deep image features using the ResUNet-based architecture.

4\. Generate a hemorrhage prediction score.

5\. Apply a classification threshold of 0.50.



Classification logic:



\- Prediction score ≥ 0.50: Hemorrhage

\- Prediction score < 0.50: Normal



The system displays the prediction result and corresponding confidence score.



\### 3.7 Stroke Risk Assessment



The final module combines the outputs of vessel narrowing detection and hemorrhage detection.



The project uses the following rule-based classification:



| Vessel Narrowing | Hemorrhage | Risk Category |

|---|---|---|

| Detected | Detected | High Risk |

| Detected | Not Detected | Moderate Risk |

| Not Detected | Detected | Moderate Risk |

| Not Detected | Not Detected | Low Risk |



The final output includes:

\- Original retinal image

\- Highlighted narrowed vessel regions

\- Hemorrhage detection result

\- Confidence score

\- Overall risk category



\## 4. Project Workflow



```text

Retinal Fundus Image

&#x20;        |

&#x20;        v

&#x20;Image Preprocessing

&#x20;        |

&#x20;        v

&#x20;Blood Vessel Segmentation

&#x20;      (U-Net)

&#x20;        |

&#x20;        v

&#x20;   Skeletonization

&#x20;        |

&#x20;        v

&#x20; Vessel Width Calculation

&#x20;        |

&#x20;        v

&#x20; Narrowing Detection

&#x20;        |

&#x20;        +----------------------+

&#x20;        |                      |

&#x20;        v                      v

&#x20;Vessel Narrowing          Hemorrhage

&#x20;   Analysis               Detection

&#x20;                             |

&#x20;                             v

&#x20;                      ResUNet Classifier

&#x20;        |                      |

&#x20;        +----------+-----------+

&#x20;                   |

&#x20;                   v

&#x20;            Risk Assessment

&#x20;                   |

&#x20;                   v

&#x20;            Result Visualization

```



\## 5. Technologies Used



| Technology | Purpose |

|---|---|

| Python | Main programming language |

| PyTorch | Deep learning model implementation |

| U-Net | Blood vessel segmentation |

| ResUNet | Hemorrhage classification |

| OpenCV | Image processing and visualization |

| NumPy | Numerical operations and feature calculations |

| Scikit-image | Skeletonization and morphological operations |

| Scikit-learn | Machine learning utilities |

| Matplotlib | Image visualization and training plots |

| Seaborn | Data visualization |

| FastAPI | Backend development and API integration |

| React.js | Frontend development |

| Git \& GitHub | Version control and project hosting |

| Git LFS | Large dataset file management |



\## 6. System Implementation



The project is implemented using Python and integrated into a web application.



\### Backend



FastAPI connects the frontend with the trained deep learning models.



It handles:

\- Retinal image upload requests.

\- Vessel segmentation.

\- Vessel narrowing analysis.

\- Hemorrhage detection.

\- Risk classification.

\- Returning analysis results through API responses.



\### Frontend



React.js is used to develop the interactive user interface.



The application includes:



\- Dashboard

\- Image Analysis

\- About Project



Users can upload and preview retinal images, initiate analysis, and view segmentation, narrowing detection, hemorrhage classification, confidence scores, and final risk categories.



\## 7. Project Structure



```text

Stroke\_Detection/

│

├── frontend/

│   └── React application

│

├── backend/

│   ├── main.py

│   ├── classification.py

│   ├── segmentation.py

│   ├── best\_resunet\_classifier.pth

│   ├── best\_vessel\_segmentation\_model\_state\_dict.pth

│   ├── narrowing\_detection\_model.pkl

│   └── requirements.txt

│

├── .gitignore

├── .gitattributes

└── README.md

```



\## 8. Installation and Setup



\### Prerequisites



\- Python

\- Node.js and npm

\- Git

\- Git LFS



\### Step 1: Clone the Repository



```bash

git clone YOUR\_GITHUB\_REPOSITORY\_URL

cd Stroke\_Prediction\_Using\_Retinal\_Images

```



\### Step 2: Download Git LFS Files



```bash

git lfs install

git lfs pull

```



\### Step 3: Backend Setup



```powershell

cd backend

python -m venv venv

.\\venv\\Scripts\\Activate.ps1

python -m pip install -r requirements.txt

```



Start the backend using the project's configured startup command.



\### Step 4: Frontend Setup



Open another terminal:



```bash

cd frontend

npm install

npm run dev

```



Open the local URL displayed in the terminal.



\## 9. Results and Analysis



The system generates outputs at different stages of image processing:



1\. Original retinal fundus image.

2\. Preprocessed retinal image.

3\. Blood vessel segmentation output.

4\. Skeletonized vessel structure.

5\. Vessel width measurements.

6\. Narrowed vessel regions highlighted in red.

7\. Hemorrhage prediction and confidence score.

8\. Final risk category.



The visual outputs help understand how retinal image features are processed and analyzed by the system.



\## 10. Dataset Management



The project uses retinal fundus datasets such as DRIVE, STARE, and CHASE\_DB1.



Large dataset archives are managed using Git Large File Storage (Git LFS).



\## 11. Future Scope



\- Improve model performance using larger and more diverse datasets.

\- Explore advanced segmentation architectures.

\- Improve vessel width measurement and narrowing analysis.

\- Evaluate the system using appropriate classification and segmentation metrics.

\- Perform further validation using clinically reviewed datasets.

\- Improve the web application's usability and visualization.



\## 12. Disclaimer



This project is developed for academic and research purposes. The generated risk categories are experimental outputs based on the implemented rules and models. They should not be considered a medical diagnosis or used as a substitute for professional medical advice.



\## 13. Author



Developed as an academic project focused on Deep Learning, Medical Image Analysis, and retinal fundus image processing.



