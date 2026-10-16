"""ค่าที่ใช้เตรียมรูปภาพ"""
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 32
SEED = 42

# แปลง RGB และปรับขนาดรูปด้วย image_dataset_from_directory
# ในตัวโมเดลใช้ Rescaling(1/255) เพื่อเปลี่ยน pixel 0-255 เป็น 0-1
