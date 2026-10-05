# Topgun: AIoT Coffee Classifier (Production Ready)

**สถานะ:** ติดตั้งและเปิดใช้งานโมเดล AI จำแนกความสุกของผลกาแฟจริงเรียบร้อยแล้ว (`mode: production`, `model_ready: true`)  
รองรับการประมวลผลบน **CPU เพียว ๆ ของ Raspberry Pi 5** โดยไม่ต้องใช้ AI Accelerator ตามข้อกำหนดของการแข่งขัน TESA Top Gun Rally #20

---

## ⚡ สเปกและประสิทธิภาพของระบบ (Live Benchmark on Pi 5)

* **สถาปัตยกรรมโมเดล:** `YOLOv11n-cls` (Exported to ONNX Runtime)
* **ขนาดโมเดล:** 5.9 MB ([models/coffee_model.onnx](models/coffee_model.onnx))
* **คลาสที่จำแนกได้ (3 ระดับ):** `ripe` (สุก / สีแดง), `semiripe` (กึ่งสุก / สีส้ม-เหลือง), `unripe` (ดิบ / สีเขียว)
* **ความแม่นยำ (Validation Accuracy):** **`96.5%`** (ประเมินบนชุดข้อมูลภาพผลกาแฟ 2,013 ตัวอย่าง)
* **เวลา Inference บน Pi 5 CPU:** **`~15.7 ms`** ⚡
* **เวลารวมทั้งระบบ (Processing Budget):** **`~30.5 ms`** (เร็วกว่างบเวลา 1,000 ms ที่โจทย์กำหนดถึง **33 เท่า**)
* **หน่วยความจำบน Pi 5:** ใช้ RAM เพียง **~565 MB จาก 8 GB** (เหลือว่าง 7.4 GB)

---

## 🚀 ลำดับงานในวันจริง (เมื่อได้ Dataset ใหม่มา หรือต้องการเทรนใหม่)

เนื่องจากการเทรนบน Mac/Laptop เร็วกว่าบน Pi มาก (1–2 นาที vs 30–60 นาที) จึงใช้ Workflow 2 สเต็ปดังนี้:

### สเต็ปที่ 1: สกัดภาพและสั่งเทรนบน Mac
```bash
# 1. สกัดภาพผลกาแฟเดี่ยวจากชุดข้อมูล
python scripts/prepare_kaggle_data.py --source /path/to/dataset --output dataset

# 2. สั่งเทรน YOLOv11n-cls พร้อมระบบ Early Stopping ป้องกัน Overfitting
python scripts/train_yolo_classifier.py --epochs 30 --patience 10 --batch-size 32
```
*ระบบจะบันทึกโมเดลใหม่ทับที่ `models/coffee_model.onnx` และ `models/labels.json` ให้อัตโนมัติ*

### สเต็ปที่ 2: ส่งไฟล์ขึ้น Pi และรีสตาร์ตใช้งานทันที (เลือกได้ 2 ทาง)

* **ทางเลือก A: ส่งตรงผ่าน LAN / Wi-Fi (แนะนำที่สุด - ทำงานได้แม้ออฟไลน์ 100%):**
  ```bash
  scp -r app.py classifier.py models web topgun@topgun-t02.local:~/topgun/ && ssh topgun@topgun-t02.local "echo topgun2026 | sudo -S systemctl restart topgun-api"
  ```
* **ทางเลือก B: ผ่าน Git บน Pi (ถ้ามีอินเทอร์เน็ต):**
  หลังจาก `git push` จากเครื่องขึ้น GitHub แล้ว สามารถสั่งอัปเดตบน Pi ในคำสั่งเดียว:
  ```bash
  ssh topgun@topgun-t02.local "~/topgun/update.sh"
  ```

---

## 📡 การเรียกใช้งาน API

### 1. ทดสอบจำแนกภาพ (`POST /predict`)
```bash
curl -X POST 'http://topgun-t02.local:8000/predict' -F 'image=@coffee.jpg'
# หรือระบุ IP: http://192.168.1.126:8000/predict
```

**ตัวอย่าง JSON ผลลัพธ์:**
```json
{
  "request_id": "aeafaa7b-a98a-4c30-a751-c5d86a37b03c",
  "status": "success",
  "label": "ripe",
  "confidence": 0.8079,
  "model_version": "v1.0-yolo11n-cls",
  "inference_ms": 15.707,
  "processing_ms": 30.589,
  "within_processing_budget": true,
  "image_width": 640,
  "image_height": 480,
  "message": "Classified successfully"
}
```
*ผลลัพธ์นี้จะถูก Publish ขึ้นหัวข้อ MQTT `topgun/coffee/results` ด้วย QoS 1 อัตโนมัติ*

### 2. ตรวจสอบความพร้อมของระบบ (`GET /ready` & `GET /health`)
```bash
curl http://topgun-t02.local:8000/ready
# ผลลัพธ์: {"transport_ready": true, "model_ready": true, "mode": "production"}

curl http://topgun-t02.local:8000/health
# ผลลัพธ์: {"status": "ok", "mode": "production", "model_ready": true, "mqtt_connected": true}
```

---

## 🖥️ การเข้าใช้งาน Web Dashboard

* เปิดผ่านเบราว์เซอร์ในวงเครือข่าย: **`http://topgun-t02.local:8000`** (หรือ `http://192.168.1.126:8000`)
* รองรับการลากวางไฟล์ภาพ (Drag & Drop)
* แสดงผลลัพธ์คลาส, ค่าความมั่นใจ (Confidence), เกจวัดเวลาประมวลผลเทียบกับงบเวลา (< 1,000 ms) และแสดงสถานะ MQTT Real-time

---

## 🧪 การทดสอบระบบ (Test Suite)

```bash
# 1. ทดสอบ Unit / Integration Tests (9 รายการ)
python -m pytest -q

# 2. ทดสอบ Edge Cases & Error Handling บน Pi จริง (7 รายการ)
python scripts/edge_test.py --url http://topgun-t02.local:8000

# 3. ทดสอบ Load Test ยิงซ้ำแบบต่อเนื่อง
python scripts/load_test.py --url http://topgun-t02.local:8000 --image test_coffee.jpg --count 20 --concurrency 4
```
