# Oil Palm Tree Detection and Segmentation using U-Net

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![scikit-image](https://img.shields.io/badge/scikit--image-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Tifffile](https://img.shields.io/badge/Tifffile-black?style=for-the-badge&logo=image&logoColor=white)
![Pickle](https://img.shields.io/badge/Pickle-grey?style=for-the-badge&logo=python&logoColor=white)
![Utils](https://img.shields.io/badge/Utils-blueviolet?style=for-the-badge)
![Sutil](https://img.shields.io/badge/Sutil-green?style=for-the-badge)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)


This project implements a deep learning model based on the **U-Net** architecture for the identification, segmentation, and counting of oil palm trees in high-resolution RGB satellite imagery. Unlike traditional methods that rely on bounding boxes, this approach utilizes pixel-level segmentation to improve precision and computational efficiency.

## 📋 Project Overview

Monitoring oil palm plantations is critical for sustainable agricultural practices, pest management, and ecosystem dynamics. This software processes large-scale orthomosaics using a **block-by-block strategy**, ensuring that high-resolution data can be processed on mid-range hardware without compromising accuracy.

**Key Features:**

* **Architecture:** Modified U-Net encoder-decoder with adjustable filter counts to optimize training time.


* **Input:** High-resolution RGB images ( m spatial resolution).


* **Output:** Segmentation masks and tree counting based on the centroid of each segmented region.



## 💻 Technical Specifications

* **Language:** Python 3.10.


* **Hardware Environment:** AMD Ryzen 5 5600x CPU, 16 GB RAM, and NVIDIA RTX 4060 GPU (8 GB VRAM).


* **Dataset:** 130-hectare plantation in Tibaitatá - Mosquera, Colombia.


* **Image Size:**  pixels.



---

## 🔬 Experiments and Results

### 1. Filter Variation ()

The study evaluated how the number of base filters affects accuracy and training time. While  achieved the highest accuracy, lower configurations (like ) offered an excellent balance for limited hardware.

![App Screenshot](img/table_1_traning.png)

### 2. Optimizer Comparison

Experiments were conducted keeping model parameters fixed to identify the most efficient optimizer.

* **For :** **RMSprop** and **Adam** reached training accuracies above 99%.


* **For :** **RMSprop** delivered the best performance with 99.74% accuracy.



### 3. Detection Performance (Testing)

Final performance was measured using Precision (P), Recall (R), and Overall Accuracy (OA).

![App Screenshot](img/table_3_testing.png)

## 🚀 Setup and Usage

1. **Clone the repository:**
```bash
https://github.com/hvargas121288/palm_oil_segmentation_satelital_images

```
2. **Install dependencies:**
```bash
pip install -r requirements.txt

```

3. **Data Preparation:** Images should be in high-resolution RGB format. The model uses a block-wise strategy to handle memory constraints.


4. **Training:** The script allows you to set the N_filt parameter to match your GPU capacity.



## 📝 Authors

* **Hector Miguel Vargas García** - *Universidad Manuela Beltran*.

*  **Ivan Fernando Bohorquez Hernandez** - *Universidad Manuela Beltran*.
  
*  **Sergio Santiago Quimbaya Rodríguez** - *Universidad Manuela Beltran*.

* **Jose Alejandro Betancur Ramirez**.

* **Ariolfo Camacho Velasco** - *AGROSAVIA*.

* **Cesar Augusto Vargas García**.


---
