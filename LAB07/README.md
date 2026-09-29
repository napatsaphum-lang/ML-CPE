# LAB07 - Convolutional Neural Network

## บทนำ

การทดลองนี้เป็นส่วนหนึ่งของรายวิชา Machine Learning โดยมีวัตถุประสงค์เพื่อศึกษาและทดลองใช้งาน Convolutional Neural Network (CNN) สำหรับงานจำแนกภาพ

ในการทดลองได้ทำการเตรียมข้อมูล แบ่งข้อมูลออกเป็น Training, Validation และ Testing จากนั้นสร้าง CNN Model เพื่อใช้ในการ Train และประเมินผลด้วย Accuracy, Loss, Confusion Matrix และ Prediction

---

## ชุดข้อมูลที่ใช้

ใช้ Dataset **Fashion MNIST PNG** จาก Kaggle

**แหล่งที่มา:**  
https://www.kaggle.com/datasets/andhikawb/fashion-mnist-png

จาก Dataset ต้นฉบับ ได้นำข้อมูลมาใช้ในการทดลองจำนวนทั้งหมด 1,000 ภาพ แบ่งเป็น 10 Classes ตั้งแต่ Class 0 ถึง Class 9

จำนวนข้อมูลที่ใช้ประกอบด้วย

- Training Dataset จำนวน 800 ภาพ
- Testing Dataset จำนวน 200 ภาพ

จาก Training Dataset จำนวน 800 ภาพ ได้แบ่งข้อมูลเพิ่มเติมสำหรับ Validation โดยแบ่งเป็น

- Training จำนวน 640 ภาพ
- Validation จำนวน 160 ภาพ
- Testing จำนวน 200 ภาพ

---

## การเตรียมข้อมูล

ก่อนนำข้อมูลเข้าสู่ CNN Model ได้ทำการเตรียมข้อมูลดังนี้

1. อ่านรูปภาพจาก Dataset
2. ปรับขนาดรูปภาพเป็น 28 x 28 Pixels
3. แปลงรูปภาพเป็น Grayscale
4. Normalize ค่า Pixel จากช่วง 0-255 ให้อยู่ในช่วง 0-1
5. แบ่งข้อมูลออกเป็น Training, Validation และ Testing

ขั้นตอนดังกล่าวช่วยให้ข้อมูลอยู่ในรูปแบบที่เหมาะสมสำหรับนำเข้า CNN Model

---

## โครงสร้าง CNN Model

CNN Model ที่ใช้ในการทดลองมีโครงสร้างดังนี้

```text
Input Image (28x28x1)
        |
        v
Conv2D
32 Filters (3x3)
        |
        v
MaxPooling2D (2x2)
        |
        v
Flatten
        |
        v
Dense 128 Neurons
        |
        v
Output 10 Classes

ค่าที่ใช้ในการ Train Model
- Optimizer: Adam
- Learning Rate: 0.0005
- Loss Function: Sparse Categorical Crossentropy
- Epochs: 10
- Batch Size: 32
ผลการทดลอง
Training and Validation Accuracy
กราฟแสดงค่า Accuracy ของ Training และ Validation ในแต่ละ Epoch
 
