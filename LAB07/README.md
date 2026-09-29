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
 
จากกราฟพบว่า Training Accuracy มีแนวโน้มเพิ่มขึ้นอย่างต่อเนื่องตามจำนวน Epoch ขณะที่ Validation Accuracy มีแนวโน้มเพิ่มขึ้นโดยรวม แม้ว่าจะมีการเปลี่ยนแปลงเล็กน้อยในบางช่วง
แสดงให้เห็นว่าโมเดลสามารถเรียนรู้จาก Training Dataset และสามารถนำรูปแบบที่เรียนรู้ไปใช้กับ Validation Dataset ได้ในระดับหนึ่ง
Training and Validation Loss
กราฟแสดงค่า Loss ของ Training และ Validation
 
จากกราฟพบว่า Training Loss ลดลงอย่างต่อเนื่องตลอดการ Train ส่วน Validation Loss มีแนวโน้มลดลงโดยรวม แต่มีการเปลี่ยนแปลงเล็กน้อยในบาง Epoch
ผลดังกล่าวแสดงให้เห็นว่าโมเดลมีการเรียนรู้ที่ดีขึ้นเมื่อจำนวน Epoch เพิ่มขึ้น
Confusion Matrix
Confusion Matrix ใช้สำหรับวิเคราะห์ผลการทำนายของโมเดลในแต่ละ Class
 
จาก Testing Dataset จำนวน 200 ภาพ โมเดลสามารถทำนายถูกได้ทั้งหมด 151 ภาพ คิดเป็น Accuracy ประมาณ 75.5%
จาก Confusion Matrix พบว่า
- Class 5 ทำนายถูก 20 จาก 20 ภาพ
- Class 9 ทำนายถูก 20 จาก 20 ภาพ
- Class 8 ทำนายถูก 19 จาก 20 ภาพ
- Class 1 ทำนายถูก 18 จาก 20 ภาพ
- Class 7 ทำนายถูก 17 จาก 20 ภาพ
ขณะที่บาง Class เช่น Class 2 และ Class 6 ยังมีการทำนายสับสนกับ Class อื่นอยู่
Prediction
ทำการสุ่มข้อมูลจาก Testing Dataset จำนวน 4 ภาพ เพื่อนำมาทดสอบกับ CNN Model ที่ Train แล้ว
 
จากตัวอย่างพบว่า
- True Class 9 → Predict Class 9
- True Class 2 → Predict Class 2
- True Class 7 → Predict Class 7
- True Class 6 → Predict Class 2
จากตัวอย่างทั้งหมด 4 ภาพ โมเดลสามารถทำนายถูก 3 ภาพ และทำนายผิด 1 ภาพ
สรุปผลการทดลอง
จากการทดลองใช้ Convolutional Neural Network สำหรับจำแนกภาพจาก Dataset Fashion MNIST PNG โดยใช้ข้อมูลจำนวนทั้งหมด 1,000 ภาพ พบว่า CNN Model สามารถเรียนรู้และจำแนกข้อมูลภาพได้
หลังจาก Train Model จำนวน 10 Epochs พบว่า Training Accuracy มีแนวโน้มเพิ่มขึ้น ขณะที่ Training Loss ลดลงอย่างต่อเนื่อง ส่วน Validation Accuracy และ Validation Loss มีแนวโน้มดีขึ้นโดยรวม
จากการทดสอบด้วย Testing Dataset จำนวน 200 ภาพ โมเดลสามารถทำนายถูก 151 ภาพ คิดเป็น Accuracy ประมาณ 75.5%
ผลการทดลองแสดงให้เห็นว่า CNN สามารถนำมาใช้สำหรับงานจำแนกภาพได้ แต่ยังมีบาง Class ที่มีลักษณะใกล้เคียงกันและทำให้เกิดความผิดพลาดในการทำนายได้
อ้างอิง
Kaggle - Fashion MNIST PNG Dataset
https://www.kaggle.com/datasets/andhikawb/fashion-mnist-png
```
