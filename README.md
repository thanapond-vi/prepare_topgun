# Topgun Coffee Classifier — Version 1.0.0

รับภาพผ่าน HTTP API → Raspberry Pi 5 รันโมเดล → Publish ผล JSON เข้า MQTT → หน้าเว็บ Subscribe และแสดงผล พร้อมตอบ JSON ทาง HTTP

## โมเดลที่ใช้ในเวอร์ชัน 1

- YOLO26n-seg RGB, best epoch35; ONNX Runtime CPU, input640×640 แบบ letterbox
- Model version: `yolo26n-seg-rgb-20261013-epoch35-onnx`
- 4 classes: `dark_roast`, `light_roast`, `medium_roast`, `unknown`
- เลือก prediction ที่มั่นใจสูงสุดเมื่อคะแนน **≥15%** ถ้าไม่มี prediction ผ่าน ตอบ `no_detection` และแสดง **ไม่พบเมล็ดกาแฟ** บนเว็บ ไม่มี mask; สถานะนี้ไม่ใช่คลาสที่5
- ไม่มีกฎสีหรือความเงามาบังคับเปลี่ยนคลาส คะแนน confidence ไม่ใช่โอกาสถูกต้องที่สอบเทียบแล้ว
- Test192 ภาพ: RGBตอบคลาสถูก184/192 (**95.83%**), box mAP50–95 **84.09%**, mask mAP50–95 **83.38%** ข้อมูลคะแนนคลาสใช้ prediction สูงสุดแบบไม่ปฏิเสธ confidence; ไม่ใช่คะแนนใหม่หลังตั้งเกณฑ์15%
- ชุด test ไม่มีภาพ null/background; ยังไม่ยืนยันการปฏิเสธพื้นหลังหรือความแม่นทุกภาพกองเมล็ด มี capture-session/near-duplicate leakage ที่ยังไม่ได้ตรวจ

Git มีโค้ด หน้าเว็บ และโมเดลที่ใช้งานจริงเพียงตัวเดียว (~11MB) ไม่รวม dataset, ref, รอบเทรน, backup, log และข้อมูล MQTT ส่วนตัว โมเดลและ SHA-256 อยู่ใน `models/coffee_segmentation.onnx` และ `models/v1.json`

## ติดตั้งเพื่อทดสอบ

ใช้ Python ที่มี wheel รองรับ ONNX Runtime บนระบบนั้น Pi เป็น Raspberry Pi OS64-bit ผู้ทดสอบต้องตั้ง MQTT ของตัวเองก่อน API จะพร้อมรับภาพ

```bash
git clone https://github.com/thanapond-vi/prepare_topgun.git
cd prepare_topgun
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
cp web/assets/mqtt-config.example.js web/assets/mqtt-config.js
```

แก้ `.env` ให้มี MQTT host, port, username และ password สำหรับ Pi Publish ผล จากนั้นแก้ `web/assets/mqtt-config.js` ให้มี WSS URL และบัญชีสำหรับเว็บ Subscribe `topgun/coffee/results` บัญชีบนเว็บเห็นได้จากเบราว์เซอร์ จึงควรให้สิทธิ์เฉพาะ Subscribe หัวข้อนี้ ทั้งสองไฟล์ถูก ignore ไม่ขึ้น Git

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

เปิด `http://localhost:8000/` จากเครื่องที่รัน หรือ IP ของเครื่องนั้น `/ready` ต้องมี `model_ready:true` และ `transport_ready:true` จึงรับภาพได้

บน Pi ที่ติดตั้งแล้วเปิด [หน้าเว็บ](http://192.168.1.126:8000/) หรือ [ชื่อ Pi](http://topgun-t02.local:8000/) IP อาจเปลี่ยนตามเครือข่าย `/widget` เป็นหน้าประกอบในระบบ

## เริ่มอัตโนมัติบน Pi

ไฟล์ service สมมติ repoอยู่ `/home/topgun/topgun` และ user=`topgun` ปรับสองค่านี้ให้ตรงเครื่องก่อนติดตั้ง

```bash
sudo cp deploy/topgun-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now topgun-api
systemctl status topgun-api --no-pager
curl -sS http://localhost:8000/ready
```

ใช้หนึ่ง worker และรองรับ inferenceครั้งละหนึ่งคำขอ คำขอพร้อมกันอาจได้429 เมื่อ MQTTหลุดจะได้503 ตรวจ `/health`, `/ready`, `/docs` สำหรับสถานะและ API

## ยิง API และวัดเวลาไป–กลับ

เปลี่ยน `/path/to/coffee.jpg` เป็นภาพของคุณ Gitไม่ได้รวม dataset/ref

```bash
curl -sS --fail-with-body \
  -o /tmp/topgun-result.json \
  -w '%{time_total}\n' \
  -F "image=@/path/to/coffee.jpg" \
  "http://192.168.1.126:8000/predict" \
  > /tmp/topgun-time.txt && python3 - <<'PYTIME'
import json
result = json.load(open('/tmp/topgun-result.json'))
total_ms = float(open('/tmp/topgun-time.txt').read()) * 1000
print('โมเดล:', result['model_version'])
print('ผล:', result['label'])
print('Predict:', result['inference_ms'], 'ms')
print('ประมวลผลบน Pi:', result['processing_ms'], 'ms')
print(f'ไป–กลับ HTTP: {total_ms:.3f} ms')
print('HTTP ไป–กลับไม่เกิน 1 วินาที:', total_ms <= 1000)
PYTIME
```

`inference_ms` วัดเฉพาะโมเดล; `processing_ms` วัดหลังรับภาพครบจนเตรียมผลทำนาย ไม่รวมอัปโหลด รอ MQTTack และส่ง response; `time_total` รวม HTTPจนรับผลครบ แต่ไม่รวมการวาดผลหน้าเว็บ `within_processing_budget` ตรวจเฉพาะเวลา server≤1000ms ไม่ใช่เวลารวมทั้งหมด

ตัวอย่าง JSON ย่อ (คะแนนและเวลาเปลี่ยนตามภาพ):

```json
{
  "status": "success",
  "label": "dark_roast",
  "confidence": 0.932014,
  "model_version": "yolo26n-seg-rgb-20261013-epoch35-onnx",
  "confidence_threshold": 0.15,
  "inference_ms": 184.743,
  "processing_ms": 217.928
}
```

ภาพไม่ผ่านเกณฑ์ได้ `label:"no_detection"`, `confidence:null`, `detection_count:0`, `mask_polygons:[]` ส่วน `unknown` เป็นคลาสที่โมเดลเรียนมา API/MQTTใช้ labelภาษาอังกฤษเพื่อเชื่อมโปรแกรม หน้าเว็บใช้ภาษาไทย

## ตรวจโค้ดและข้อมูลการทดลอง

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

[ผลเปรียบเทียบ RGB/BGR](reports/rgb-bgr-comparison-20261013/README.md), [รอบทดลองและการเทรน](docs/training-experiments.md), [Pi deployment](docs/pi-segmentation.md) เป็นหลักฐานและประวัติ ภาพ plots/dataset/logไม่รวม Git ต้องเตรียม datasetและbaselineท้องถิ่นก่อนใช้สคริปต์เทรน สคริปต์และข้อมูลการเทรนไม่จำเป็นสำหรับ inference

---

## 📚 แหล่งอ้างอิง Dataset

* **ชุดข้อมูล:** [Coffee Bean Dataset (Resized 224x224)](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224) โดย Gerry Piosenka บน Kaggle
* **ชุดข้อมูลกองเมล็ดพร้อม metadata Agtron:** [Coffee Toasted Agtron](https://www.kaggle.com/datasets/joaovvrodrigues/coffee-toasted-agtron) โดย joaovvrodrigues ต้องตรวจสเกลและวิธีวัดของต้นทางก่อนนำค่าไปใช้
* **ชุดข้อมูลกองเมล็ดเพิ่มเติม:** [Coffee Bean Roast Degree](https://universe.roboflow.com/coffee-bean-roast-degree/coffee-bean-roast-degree) บน Roboflow Universe, CC BY 4.0; labels ต้นทางเป็น medium/not-medium ต้องตรวจและทำ polygon labels ใหม่สำหรับ 4 คลาสของโปรเจกต์

## เกณฑ์เลือกระดับการคั่วและแหล่งอ้างอิง

### เกณฑ์ของโปรเจกต์สำหรับข้อมูลที่มีค่า Agtron

ผู้จัดทำกำหนดจุดตัดดังต่อไปนี้เมื่อวันที่ 12 ตุลาคม 2026 เพื่อรวมระดับการคั่วเป็น 3 คลาส ช่วงเหล่านี้เป็น **เกณฑ์ของโปรเจกต์** ที่พิจารณาจากข้อมูลอ้างอิงด้านล่าง ไม่ใช่ตารางจุดตัดที่คัดลอกจากมาตรฐาน SCA หรือ Suzuki โดยตรง

| คลาส | ช่วง Agtron ที่โปรเจกต์เลือก | ค่าใน Coffee Toasted Agtron ที่เข้ากลุ่ม |
|---|---|---|
| `light_roast` (คั่วอ่อน) | 65 ≤ A ≤ 100 | 65, 75 |
| `medium_roast` (คั่วกลาง) | 45 ≤ A < 65 | 45, 55 |
| `dark_roast` (คั่วเข้ม) | 0 ≤ A < 45 | 25, 35 |

ใช้ช่วงต่อเนื่องเพื่อรองรับค่าทศนิยมโดยไม่มีช่องว่าง ค่า 45 เป็น medium และค่า 65 เป็น light ส่วน `unknown` เป็นคลาสแยกสำหรับลักษณะเมล็ดที่อยู่นอกกลุ่มที่กำหนด เช่น เมล็ดเขียว ไม่ได้กำหนดจากช่วง Agtron นี้ ค่าที่หายไปหรืออยู่นอก 0–100 ให้ตรวจข้อมูลก่อน ไม่แปลงเป็น unknown อัตโนมัติ ขอบเขต 0 และ 100 ไม่ใช่หลักฐานว่าชุดข้อมูลมีตัวอย่างที่ปลายช่วงเหล่านั้น

### ข้อมูลสากลและสิ่งที่แหล่งอ้างอิงระบุจริง

Agtron เป็นระบบวัดระดับสีการคั่วที่ใช้ในอุตสาหกรรมกาแฟ ค่าต่ำสัมพันธ์กับสีเข้มกว่า ต้องแยก Gourmet Scale กับ Commercial Scale และแยกการวัดเมล็ดทั้งเมล็ดกับกาแฟบด เพราะตัวเลขไม่สามารถเทียบกันตรง ๆ โดยละเลยวิธีวัดได้ [Rubasse Roasters: Agtron Color Measurement](https://rubasseroasters.com/agtron-color-measurement-the-most-comprehensive-guide-ever/) อธิบายความแตกต่างเหล่านี้ รวมถึงผลของการเตรียมตัวอย่างต่อค่าที่อ่านได้ บทความนี้เป็นข้อมูลจากผู้ผลิตเครื่องคั่ว ไม่ใช่เอกสารมาตรฐานที่ SCA ออกเอง

[Suzuki Coffee Thailand: การวัดระดับสีการคั่วกาแฟ](https://suzuki-coffee.com/การวัดระดับสีเมล็ดกาแฟ/) ระบุว่าเรียบเรียงจากข้อมูล SCAA และให้ช่วงดังนี้:

| ระดับในบทความ Suzuki | Agtron |
|---|---|
| Very Light | 80–90 |
| Light | 70–80 |
| Medium Light | 60–70 |
| Medium | 50–60 |
| Medium-Dark | 40–50 |
| Dark | 30–40 |
| Very Dark | 25–30 |

ตัวเลขนี้ใช้ประกอบเหตุผลในการเลือกเกณฑ์ แต่จุดตัด 45 และ 65 ของโปรเจกต์เป็นการตัดสินใจของผู้จัดทำ เช่น 45 ถูกจัดเป็น medium ในโปรเจกต์ แม้บทความ Suzuki จัดช่วง 40–50 เป็น Medium-Dark

แหล่งมาตรฐานโดยตรงคือ [Specialty Coffee Association: Standards](https://sca.coffee/research/coffee-standards) ณ วันที่ตรวจสอบ 12 ตุลาคม 2026 หน้าเว็บระบุ **SCA-131 Coffee Roast Level: Test Method** อยู่ในหมวด Standards in Development หน้านี้ไม่ได้ยืนยันจุดตัด 45/65 ของโปรเจกต์ การมีสเกลหรือแผ่นเทียบสีอ้างอิงสากลจึงต้องแยกจากการรวมเป็น 3 คลาสสำหรับงานนี้

### ข้อจำกัดของ labels ชุดเมล็ดเดี่ยวและ baseline

การทำนายปัจจุบันใช้คะแนนคลาสที่โมเดลเรียนจากภาพและ labels ใน dataset แล้วเลือก prediction ที่มั่นใจสูงสุด ไม่มีการกำหนดกฎว่าสีครีม/เหลือง/น้ำตาลต้องเป็นคั่วอ่อน หรือใช้ความเงาปรับคลาส และไม่มีการคำนวณ Agtron จากภาพเพื่อบังคับผลลัพธ์ ช่วง Agtron ด้านบนใช้กำหนด labels ของข้อมูลที่มี metadata เท่านั้น

ชุด Coffee Bean Dataset ใช้ชื่อคลาสจากเจ้าของ dataset โปรเจกต์ยังไม่มีค่า Agtron ที่วัดรายภาพหรือหลักฐานว่าจุดตัดของเจ้าของตรงกับเกณฑ์ใหม่ จึงไม่กำหนดค่า Agtron ย้อนกลับจากชื่อคลาสหรือสีในภาพ baseline YOLO26n-seg epoch 25 เรียนตาม labels ชุดเดิม และยังไม่ผ่านการยืนยันตามช่วง Agtron ที่กำหนดใหม่นี้

ก่อนรวมข้อมูลเมล็ดเดี่ยวกับชุด Agtron ให้ตรวจความสอดคล้องของ labels เก็บที่มาของ label และค่า Agtron แยกกัน และพักภาพก้ำกึ่งไว้ตรวจ การแก้ README นี้ไม่ได้เปลี่ยน labels dataset หรือเทรนโมเดลใหม่
