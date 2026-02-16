# Oil Palm Tree Detection and Segmentation using U-Net

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-blue?style=for-the-badge)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
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


* **Dataset:** 130-hectare plantation in Caquetá, Colombia.


* **Image Size:**  pixels.



---

## 🔬 Experiments and Results

### 1. Filter Variation ()

The study evaluated how the number of base filters affects accuracy and training time. While  achieved the highest accuracy, lower configurations (like ) offered an excellent balance for limited hardware.

|  | Training Acc. | Total Time (s) | Trainable Parameters |
| --- | --- | --- | --- |
| 4 | 0.984 

 | 1502.8 

 | 121,725 

 |
| 16 | 0.996 

 | 4369.5 

 | 1,941,105 

 |
| 32 | 0.998 

 | 8980.8 

 | 7,760,097 

 |

### 2. Optimizer Comparison

Experiments were conducted keeping model parameters fixed to identify the most efficient optimizer.

* **For :** **RMSprop** and **Adam** reached training accuracies above 99%.


* **For :** **RMSprop** delivered the best performance with 99.74% accuracy.



### 3. Detection Performance (Testing)

Final performance was measured using Precision (P), Recall (R), and Overall Accuracy (OA).

|  | Precision (P) | Recall (R) | OA (Overall) |
| --- | --- | --- | --- |
| 6 | 93.9% 

 | 99.1% 

 | 96.5% 

 |
| 8 | 94.9% 

 | 99.6% 

 | 97.2% 

 |
| 16 | 95.5% 

 | 99.4% 

 | 97.5% 

 |

---

## 🚀 Setup and Usage

1. **Clone the repository:**
```bash
git clone https://github.com/hvargas121288/oil-palm-detection.git

```


2. **Data Preparation:** Images should be in high-resolution RGB format. The model uses a block-wise strategy to handle memory constraints.


3. 
**Training:** The script allows you to set the `$N_{filt}$` parameter to match your GPU capacity.



## 📝 Authors

* **Hector Miguel Vargas García** - *Universidad Manuela Beltran*.


* **Jose Alejandro Betancur Ramirez**.


* **Ivan Fernando Bohorquez Hernandez** - *Universidad Manuela Beltran*.


* **Ariolfo Camacho Velasco** - *AGROSAVIA*.


* **Cesar Augusto Vargas García**.


---
