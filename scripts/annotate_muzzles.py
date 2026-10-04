"""Local muzzle annotation UI. Requires Pillow; binds only to localhost."""
import csv
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'outputs/image_audit/muzzle_annotations.csv'
DATA = ROOT / 'data/clean'
LOCK = threading.Lock()
HTML = ROOT / 'src/annotation.html'

def read_rows():
    with CSV.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def image_path(row):
    p = (DATA / row['image_path']).resolve()
    if not p.is_relative_to(DATA.resolve()): raise ValueError('Invalid image path')
    return p

class Handler(BaseHTTPRequestHandler):
    def reply(self, body, mime='application/json', status=200):
        if not isinstance(body, bytes): body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == '/': return self.reply(HTML.read_bytes(), 'text/html; charset=utf-8')
            with LOCK: rows = read_rows()
            if parsed.path == '/api/images': return self.reply(rows)
            if parsed.path == '/api/image':
                i = int(parse_qs(parsed.query)['index'][0])
                if not 0 <= i < len(rows): raise ValueError('Invalid index')
                with Image.open(image_path(rows[i])) as im:
                    rgb = ImageOps.exif_transpose(im).convert('RGB')
                    buffer = io.BytesIO(); rgb.save(buffer, format='JPEG', quality=95)
                return self.reply(buffer.getvalue(), 'image/jpeg')
            self.reply({'error':'Not found'}, status=404)
        except Exception as e: self.reply({'error':str(e)}, status=400)

    def do_POST(self):
        # Prevent cross-origin browser writes to the local annotation service.
        if self.headers.get('Origin') != 'http://127.0.0.1:8765':
            return self.reply({'error':'Invalid origin'}, status=403)
        if self.path != '/api/annotation': return self.reply({'error':'Not found'},status=404)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4096: raise ValueError('Invalid request size')
            payload = json.loads(self.rfile.read(length))
            with LOCK:
                rows = read_rows(); i = int(payload['index'])
                if not 0 <= i < len(rows): raise ValueError('Invalid index')
                row = rows[i]; status = payload['status']
                if status not in {'yes','no'}: raise ValueError('Invalid annotation status')
                if status == 'yes':
                    box = payload['box']
                    if len(box) != 4 or any(type(v) is not int for v in box): raise ValueError('Integer coordinates required')
                    x1,y1,x2,y2 = box
                    with Image.open(image_path(row)) as im: w,h = ImageOps.exif_transpose(im).size
                    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h): raise ValueError('Box outside image')
                    if min(x2-x1,y2-y1) < 32: raise ValueError('Box too small; inspect muzzle quality')
                    for key,value in zip(('x_min','y_min','x_max','y_max'),box): row[key]=str(value)
                else:
                    for key in ('x_min','y_min','x_max','y_max'): row[key]=''
                row['muzzle_usable']=status
                temp = CSV.with_suffix('.tmp')
                with temp.open('w',newline='',encoding='utf-8') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                os.replace(temp,CSV)
            self.reply({'saved':True})
        except Exception as e: self.reply({'error':str(e)},status=400)

if __name__ == '__main__':
    print('Muzzle annotation: http://127.0.0.1:8765',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8765),Handler).serve_forever()
