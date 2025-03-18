# Bonefracturedtection-using-deep-learning-

```md
# 🦴 Bone Fracture Detection Using Deep Learning  

## 📌 Overview  
This project utilizes advanced deep learning architectures to detect bone fractures from X-ray images with **98% accuracy**. By leveraging **R2U-Net, U-Net, Vision Transformer, and Attention U-Net**, the model efficiently segments and classifies fractures for accurate diagnosis.  

## 🚀 Features  
- **Multi-model approach**: R2U-Net, U-Net, Vision Transformer, Attention U-Net  
- **98% accuracy** on the test dataset  
- **Automatic X-ray image segmentation and fracture detection**  
- **Attention-based models** for improved localization  
- **Trained on large-scale medical datasets**  

## 🛠️ Technologies Used  
- **Deep Learning Frameworks:** TensorFlow, PyTorch  
- **Model Architectures:**  
  - **U-Net & R2U-Net** (for segmentation)  
  - **Vision Transformer (ViT)** (for feature extraction)  
  - **Attention U-Net** (for improved focus on fracture regions)  
- **Computer Vision:** OpenCV  
- **Dataset:** Publicly available medical X-ray datasets  

## 📂 Dataset  
We trained and validated our models using medical X-ray datasets, such as:  
- **MURA (Musculoskeletal Radiographs Dataset)**  
- **RSNA Bone Age dataset**  
- **Custom annotated dataset** for fractures  

## 🏗️ Installation & Setup  
1. Clone the repository:  
   ```bash
   git clone https://github.com/Mahmamad-Rafi/Bonefracturedetection-using-deep-learning-.git
   cd Bonefracturedetection-using-deep-learning-
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run inference on an X-ray image:  
   ```bash
   python predict.py --image sample_xray.jpg
   ```

## 📊 Model Performance  
| Model            | Accuracy | Dice Score | Jaccard Index |
|-----------------|---------|------------|---------------|
| **U-Net**       | 94%     | 0.87       | 0.82          |
| **R2U-Net**     | 96%     | 0.91       | 0.87          |
| **Vision Transformer** | 97% | 0.93       | 0.89          |
| **Attention U-Net** | 98% | 0.95       | 0.92          |

## 🤖 Future Enhancements  
- Fine-tuning with **larger datasets**  
- **Real-time fracture detection** in mobile applications  
- Deployment as a **web-based diagnosis tool**  

## 📜 License  
This project is for research and educational purposes.  

