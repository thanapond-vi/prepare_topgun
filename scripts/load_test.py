"""Run on the client: one image/request, repeated with bounded concurrency."""
import argparse
import concurrent.futures
import json
import mimetypes
import statistics
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True, help='Base URL, e.g. http://192.168.1.126:8000')
    parser.add_argument('--image', required=True)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--output', default='load-results.json')
    args = parser.parse_args()
    if args.count < 1 or not 1 <= args.concurrency <= 32:
        parser.error('count must be positive and concurrency must be 1..32')
    path = Path(args.image)
    image = path.read_bytes()
    boundary = uuid.uuid4().hex
    mime = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="test-image"\r\n'
            f'Content-Type: {mime}\r\n\r\n').encode() + image + f'\r\n--{boundary}--\r\n'.encode()

    def send(index):
        start = time.perf_counter()
        request = urllib.request.Request(args.url.rstrip('/') + '/predict', data=body, headers={
            'Content-Type': 'multipart/form-data; boundary=' + boundary}, method='POST')
        code = None
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                code, payload = response.status, response.read()
        except urllib.error.HTTPError as error:
            code, payload = error.code, error.read()
        except Exception as error:
            return {'index': index, 'http_status': None, 'error': str(error),
                    'end_to_end_ms': round((time.perf_counter()-start)*1000, 3)}
        try:
            result = json.loads(payload)
        except ValueError:
            result = {'error': 'Non-JSON response', 'preview': payload[:200].decode(errors='replace')}
        return {'index': index, 'http_status': code, 'response': result,
                'end_to_end_ms': round((time.perf_counter()-start)*1000, 3)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(send, range(args.count)))
    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    ok = [r for r in results if r['http_status'] == 200]
    times = sorted(r['end_to_end_ms'] for r in results)
    print(json.dumps({'requests':len(results), 'http_200':len(ok), 'errors':len(results)-len(ok),
                      'end_to_end_mean_ms':round(statistics.mean(times),3),
                      'end_to_end_max_ms':max(times),
                      'processing_over_1000_ms':sum(
                          r.get('response',{}).get('processing_ms',0)>1000 or
                          (r.get('response',{}).get('detail',{}).get('processing_ms',0)>1000
                          if isinstance(r.get('response',{}).get('detail'), dict) else False)
                          for r in results),
                      'note':'Transport test only. Errors/429 do not count as passing classification.'}, indent=2))


if __name__ == "__main__":
    main()
