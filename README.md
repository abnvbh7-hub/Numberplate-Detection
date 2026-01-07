## 🚗 Automatic Number Plate Detection, Tracking & OCR

This project performs **real-time vehicle number plate detection, tracking, and text extraction** from video. It combines object detection (**YOLO**), multi-object tracking (**DeepSORT**), and optical character recognition (**EasyOCR**) into one end-to-end pipeline.

### 🖼️ Project Output:

The system generates:

#Sample Inference 1:

![Screenshot_20260107-081534_Video Player](https://github.com/user-attachments/assets/5bf5efeb-cfe3-40f8-82a1-d3dad7198572)

#Sample Inference 2:

![Screenshot_20260107-081527_Video Player](https://github.com/user-attachments/assets/c5a27e33-6cd1-4c98-8e04-0c6e0396d3f0)

#Sample Inference 3:

![Screenshot_20260107-081545_Video Player](https://github.com/user-attachments/assets/61faddc5-afbe-457d-aa7b-66c524659e14)

#Sample Inference 4:

![Screenshot_20260107-081551_Video Player](https://github.com/user-attachments/assets/4256f3fa-127a-4ef4-b4e8-4c191b7950fe)

#Sample Inference 5:

![Screenshot_20260107-081517_Video Player](https://github.com/user-attachments/assets/80d2c06f-dc6a-49c9-91dc-f9ccba209b51)



---

### 🎯 What this project does

- 🔍 Detects number plates in video frames using YOLO  
- 🧭 Tracks vehicles and plates across frames using DeepSORT  
- 🧾 Crops detected plate regions automatically  
- 🔤 Extracts text from plates using EasyOCR  
- 🎥 Generates an annotated output video with IDs and bounding boxes  

---

### 🧠 Tech Stack

- **Python**
- **OpenCV**
- **Ultralytics YOLO**
- **DeepSORT**
- **EasyOCR**
- **Plotly** (optional visualizations)

---

### 🛠️ How it works

1. Load trained YOLO model (`best.pt`)  
2. Stream frames from input video   
3. Detect number plates in each frame  
4. Track them using unique DeepSORT IDs  
5. Crop each detected plate  
6. Run OCR to extract text  
7. Save final processed video 

---

### 📦 Main Features

- ✔️ End-to-end automated ALPR-style pipeline  
- ✔️ Works on recorded videos  
- ✔️ Reusable OOP-based video handler class  
- ✔️ Confidence-based detection filtering  
- ✔️ Output video with boxes + track IDs  

---




