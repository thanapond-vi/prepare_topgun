# Topgun: API + MQTT (ยังไม่ติดตั้งโมเดล)

สถานะ: โค้ดสำหรับทดสอบการเชื่อมระบบ ไม่ใช่ระบบจำแนกที่พร้อมตรวจคะแนน
`POST /predict` รับ multipart field `image`, ตอบ JSON และ Publish JSON เดียวกันไป
`topgun/coffee/results` ด้วย QoS 1 ผ่าน TLS โดยคงการเชื่อม MQTT และ reconnect อัตโนมัติ
ผลระบุ `status: test`, `label: test_only`, `inference_ms: null`, `model_version: null` เสมอ

## ติดตั้งบน Pi ที่มี .env เดิม

บน Mac ส่งโค้ดเข้าที่พักก่อน (เปลี่ยน IP ถ้าเปลี่ยนเครือข่าย):

```bash
ssh topgun@192.168.1.126 'mkdir -p ~/topgun-update'
scp -r app.py requirements.txt web deploy scripts topgun@192.168.1.126:~/topgun-update/
```

บน Pi: กด Ctrl+C หยุด uvicorn เดิม แล้วรัน:

```bash
cd ~/topgun
cp app.py "app.py.backup-$(date +%Y%m%d-%H%M%S)"
cp ~/topgun-update/app.py ~/topgun-update/requirements.txt .
cp -R ~/topgun-update/web ~/topgun-update/deploy ~/topgun-update/scripts .
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m py_compile app.py
sudo install -m 644 deploy/topgun-api.service /etc/systemd/system/topgun-api.service
sudo systemctl daemon-reload
sudo systemctl enable --now topgun-api
sudo systemctl restart topgun-api
```

ไม่คัดลอก .env จาก Mac ไม่ต้องรัน enable_web.py อีก รุ่นนี้มีหน้าเว็บแล้ว
service template ใช้ user `topgun` และ `/home/topgun/topgun` ตามเครื่องที่ใช้อยู่
หากชื่อผู้ใช้/ตำแหน่งต่างออกไป ต้องแก้ service ก่อนติดตั้ง

```bash
systemctl status topgun-api --no-pager
curl http://localhost:8000/ready
journalctl -u topgun-api -n 40 --no-pager
```

`/health` เป็น liveness; `/ready` ตอบ 200 เมื่อ MQTT เชื่อมแล้ว แต่ยังระบุ
`model_ready: false` เพราะไม่ได้โหลดโมเดล การทดสอบการติดตั้งต้องทำบน Pi จริง
รีบูตแล้วทดสอบ `/ready` และส่งภาพโดยไม่เปิด SSH ไปสั่งบริการอีกครั้ง

## API ที่ส่งให้อาจารย์

```bash
curl -X POST 'http://topgun-t02.local:8000/predict' -F 'image=@coffee.jpg'
# หรือระบุ IP: http://192.168.1.126:8000/predict
```

ส่งทีละภาพต่อ request ได้ ใช้ request_id จาก JSON จับคู่กับ MQTT
ชื่อไฟล์/ภาพไม่ได้ถูกบันทึกลงดิสก์ถาวรโดยแอป (multipart อาจใช้ไฟล์ชั่วคราว)
ขนาดไฟล์สูงสุด 10 MB, รูปสูงสุด 20 ล้านพิกเซล, รองรับไม่เกิน 4 งานในส่วนประมวลผล/ส่ง MQTT พร้อมกัน
ใช้ uvicorn worker เดียวตาม service เพื่อให้ขีดจำกัดตรงกัน

- 200: ทดสอบการขนส่งข้อมูลสำเร็จและได้รับ MQTT QoS 1 acknowledgement จาก broker
  ไม่ใช่การยืนยันว่าเว็บได้รับข้อความหรือจำแนกถูกต้อง
- 400/413/422: อินพุตไม่ถูกต้อง/ใหญ่เกิน/ไม่มีภาพ
- 429: งานเต็ม ให้รออย่างน้อย Retry-After ก่อนลองใหม่ นับเป็นงานที่ไม่ผ่านการตรวจครั้งนั้น
- 503: broker ไม่พร้อม หรือยังยืนยันการส่งไม่ได้ภายใน 3 วินาที
  ข้อความที่รอส่งอาจยังถึง MQTT ภายหลัง ให้จับคู่ request_id และไม่ใช้ผลล่าช้าเป็นงานใหม่
- 504: processing เกิน 1 วินาที ไม่ถือว่าสำเร็จ

processing_ms เริ่มจากรับ HTTP body ครบ รวมการ parse multipart หลังรับครบ
การรอจัดตารางทำงาน และตรวจ/ถอดรหัสภาพ จนเตรียมผลพร้อมส่ง
ไม่รวมเวลาอัปโหลดและรอ MQTT acknowledgement; เวลารวมที่ผู้เรียกเห็นต้องวัดแยก
ค่าตอนนี้ไม่มี inference จึงไม่ใช่หลักฐานว่ารุ่นที่มีโมเดลผ่าน 1 วินาที
ส่วน parse/upload ยังอยู่นอกขีดจำกัด 4 งาน; นี่ไม่ใช่ขีดจำกัดทรัพยากรทั้ง HTTP server

## ทดสอบยิงซ้ำจาก Mac

ใช้ภาพของตัวเองและ URL ที่ต้องการทดสอบ สคริปต์ใช้ Python standard library:

```bash
python3 scripts/load_test.py --url http://192.168.1.126:8000 --image /path/to/coffee.jpg --count 20 --concurrency 1 --output load-results-sequential.json
python3 scripts/load_test.py --url http://192.168.1.126:8000 --image /path/to/coffee.jpg --count 20 --concurrency 4 --output load-results-parallel.json
```

บันทึก HTTP status, JSON, request_id และ end-to-end latency ทุกรายการ
นับ error และ 429 แยก ห้ามรายงานเฉพาะค่าเฉลี่ยของงานที่สำเร็จ
ทดสอบเพิ่มด้วยภาพหลายขนาด/หลายชนิดก่อนส่งงาน

## การเชื่อมต่อและการเข้าใช้งานเว็บ

- เปิดเข้าหน้าเว็บได้โดยตรงผ่าน LAN: `http://topgun-t02.local:8000` (หรือ `http://192.168.1.126:8000`)
- หน้าเว็บทำการเชื่อมต่อ MQTT Broker ให้อัตโนมัติด้วยบัญชี `topgun-web` (มีสิทธิ์เฉพาะ Subscribe รับผล)
- มี Drag & Drop รองรับการลากวางไฟล์ภาพ พร้อมแสดงสถานะเวลาประมวลผลและการตรวจสอบงบเวลา (< 1,000 ms) ทันที
- เว็บใช้ MQTT.js ผ่าน WSS จาก CDN จึงต้องมีอินเทอร์เน็ตในการโหลด library และเชื่อม broker
- API สำหรับเรียกภายนอกในวงเครือข่ายสามารถเรียกตรงไปยัง `http://topgun-t02.local:8000/predict` หรือ IP ของบอร์ดได้โดยตรง

## ตรวจโค้ดบนเครื่องพัฒนา

```bash
python -m pip install -r requirements.txt pytest httpx
python -m pytest -q
```

ชุดทดสอบใช้ broker จำลอง ไม่เชื่อม EMQX จริง ไม่อ่านหรือแสดง credentials ใน output
การยืนยัน performance/reboot/MQTT บน Pi ต้องทดสอบฮาร์ดแวร์จริงต่างหาก
