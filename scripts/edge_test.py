"""Automated edge-case and error-handling test suite for Topgun API."""
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from PIL import Image

BASE_URL = os.environ.get('TOPGUN_URL', 'http://192.168.1.126:8000').rstrip('/')

def send_multipart(endpoint, filename, content_type, data):
    boundary = uuid.uuid4().hex
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f'Content-Type: {content_type}\r\n\r\n'
    ).encode() + data + f'\r\n--{boundary}--\r\n'.encode()

    req = urllib.request.Request(
        f'{BASE_URL}{endpoint}',
        data=body,
        headers={
            'Content-Type': f'multipart/form-data; boundary={boundary}'
        },
        method='POST'
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            elapsed = (time.perf_counter() - start) * 1000
            return resp.status, resp.read().decode('utf-8', errors='replace'), elapsed
    except urllib.error.HTTPError as e:
        elapsed = (time.perf_counter() - start) * 1000
        return e.code, e.read().decode('utf-8', errors='replace'), elapsed
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return None, str(e), elapsed

def test_case(name, func):
    print(f"▶ Testing: {name} ... ", end='', flush=True)
    try:
        passed, msg = func()
        if passed:
            print(f"✅ PASS ({msg})")
            return True
        else:
            print(f"❌ FAIL: {msg}")
            return False
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False

def test_valid_image():
    out = io.BytesIO()
    Image.new('RGB', (100, 100), color='brown').save(out, format='JPEG')
    code, resp, ms = send_multipart('/predict', 'coffee.jpg', 'image/jpeg', out.getvalue())
    if code == 200:
        data = json.loads(resp)
        return True, f"Status 200, latency {ms:.1f}ms, request_id={data.get('request_id')[:8]}..."
    return False, f"Expected 200, got {code}: {resp}"

def test_text_file_instead_of_image():
    # Sending plain text content instead of a real image
    text_data = b"This is clearly a text file, not a coffee bean picture."
    code, resp, ms = send_multipart('/predict', 'test.txt', 'text/plain', text_data)
    if code == 400:
        return True, f"Rejected with 400 Bad Request as expected ({ms:.1f}ms)"
    return False, f"Expected 400, got {code}: {resp}"

def test_corrupted_image_header():
    # JPEG header with corrupted/truncated body
    corrupt_jpeg = b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00DEADBEEF'
    code, resp, ms = send_multipart('/predict', 'corrupt.jpg', 'image/jpeg', corrupt_jpeg)
    if code == 400:
        return True, f"Rejected with 400 Bad Request as expected ({ms:.1f}ms)"
    return False, f"Expected 400, got {code}: {resp}"

def test_empty_file():
    empty_data = b""
    code, resp, ms = send_multipart('/predict', 'empty.jpg', 'image/jpeg', empty_data)
    if code == 400:
        return True, f"Rejected with 400 Bad Request as expected ({ms:.1f}ms)"
    return False, f"Expected 400, got {code}: {resp}"

def test_oversized_file():
    # Generating 10.5 MB payload (exceeding 10 MB limit)
    oversized = b'A' * (10 * 1024 * 1024 + 512 * 1024)
    code, resp, ms = send_multipart('/predict', 'oversize.jpg', 'image/jpeg', oversized)
    if code == 413:
        return True, f"Rejected with 413 Payload Too Large as expected ({ms:.1f}ms)"
    return False, f"Expected 413, got {code}: {resp}"

def test_missing_image_field():
    # POST without image field
    req = urllib.request.Request(f'{BASE_URL}/predict', data=b"{}", headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            code = resp.status
    except urllib.error.HTTPError as e:
        code = e.code
    except Exception as e:
        return False, str(e)
    if code == 422:
        return True, "Rejected with 422 Unprocessable Entity (Missing Form Field)"
    return False, f"Expected 422, got {code}"

def test_health_check():
    req = urllib.request.Request(f'{BASE_URL}/health', method='GET')
    with urllib.request.urlopen(req, timeout=10) as resp:
        if resp.status == 200:
            data = json.loads(resp.read().decode())
            return True, f"Health OK, mqtt_connected={data.get('mqtt_connected')}"
    return False, "Health check failed"

def main():
    print("=" * 60)
    print(f"Topgun Edge-Case & Error-Handling Test on: {BASE_URL}")
    print("=" * 60)
    cases = [
        ("Health & MQTT Check", test_health_check),
        ("Standard Valid Image (JPEG)", test_valid_image),
        ("Sending Text File (.txt)", test_text_file_instead_of_image),
        ("Corrupted/Truncated JPEG Data", test_corrupted_image_header),
        ("Empty File (0 bytes)", test_empty_file),
        ("Oversized File (> 10 MB)", test_oversized_file),
        ("Missing Required Form-Data", test_missing_image_field),
    ]

    passed_count = 0
    for name, func in cases:
        if test_case(name, func):
            passed_count += 1

    print("=" * 60)
    print(f"Results: {passed_count}/{len(cases)} passed.")
    if passed_count == len(cases):
        print("🎉 All Edge Cases Passed! API is robust against errors.")
    else:
        print("⚠️ Some tests failed. Please inspect logs.")
    print("=" * 60)

if __name__ == '__main__':
    main()
