import os
import mimetypes
from flask import Flask, request, jsonify
from flask_cors import CORS
from helpers import load_model, process_image, process_video
from urllib.parse import quote
from google.cloud import storage
from google.auth.transport.requests import AuthorizedSession
from google.resumable_media import requests, common

os.environ['GOOGLE_CLOUD_PROJECT'] = 'roads-404204'
app = Flask(__name__)
# Define CORS policy
CORS(
    app,
    supports_credentials=True,
    origins=[
        'http://localhost:3000',
        'http://localhost:3001',
        'https://testingapirdd.x-camp.id',
        'https://roadinspecx.x-camp.id'
    ],
    methods=['GET', 'POST'],
    allowed_headers=['Content-Type', ''],
)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3600  # Set the timeout to 1 hour (in seconds)
app.config['MAX_CONTENT_LENGTH'] = 2147483648   # 2GB limit
bucket_name = 'asia.artifacts.roads-404204.appspot.com'

class GCSObjectStreamUpload(object):
    def __init__(
            self, 
            client: storage.Client,
            bucket_name: str,
            blob_name: str,
            chunk_size: int=256 * 1024
        ):
        self._client = client
        self._bucket = self._client.bucket(bucket_name)
        self._blob = self._bucket.blob(blob_name)
        self._buffer = b''
        self._buffer_size = 0
        self._chunk_size = chunk_size
        self._read = 0
        self._transport = AuthorizedSession(
            credentials=self._client._credentials
        )
        self._request = None  # type: requests.ResumableUpload

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.stop()
            
    def start(self):
        url = (
            f'https://www.googleapis.com/upload/storage/v1/b/'
            f'{self._bucket.name}/o?uploadType=resumable'
        )
        self._request = requests.ResumableUpload(
            upload_url=url, chunk_size=self._chunk_size
        )
        self._request.initiate(
            transport=self._transport,
            content_type='application/octet-stream',
            stream=self,
            stream_final=False,
            metadata={'name': self._blob.name},
        )
        
    def stop(self):
        self._request.transmit_next_chunk(self._transport)
        
    def write(self, data: bytes) -> int:
        data_len = len(data)
        self._buffer_size += data_len
        self._buffer += data
        del data
        while self._buffer_size >= self._chunk_size:
            try:
                self._request.transmit_next_chunk(self._transport)
            except common.InvalidResponse:
                self._request.recover(self._transport)
        return data_len
    
    def read(self, chunk_size: int) -> bytes:
        # I'm not good with efficient no-copy buffering so if this is
        # wrong or there's a better way to do this let me know! :-)
        to_read = min(chunk_size, self._buffer_size)
        memview = memoryview(self._buffer)
        self._buffer = memview[to_read:].tobytes()
        self._read += to_read
        self._buffer_size -= to_read
        return memview[:to_read].tobytes()
    def tell(self) -> int:
        return self._read

@app.route('/detect_image/<image_filename>', methods=['GET', 'POST'])
def detect_image(image_filename):
    # Define the path for saving the output image and JSON
    base_path, _ = os.path.splitext(image_filename)
    image_output_path = os.path.join('image-detection', f'{base_path}_output{_}')

    model = load_model()
    objects_detected_data = process_image(model, image_filename, image_output_path)
    image_output_link = objects_detected_data['image_output_link']
    objects_detected = objects_detected_data['objects_detected']
    json_output_link = objects_detected_data['json_output_link']

    response = {
        'image_output_link': image_output_link,
        'objects_detected': objects_detected,
        'json_output_link': json_output_link,
    }
    return jsonify(response)

@app.route('/detect_video/<video_filename>', methods=['GET', 'POST'])
def detect_video(video_filename):
    # Define the path for saving the annotated video and JSON
    base_path, _ = os.path.splitext(video_filename)
    output_suffix = '_output'
    
    video_output_path = os.path.join('video-detection', f'{base_path}{output_suffix}{_}')

    model = load_model()
    objects_detected_data = process_video(model, video_filename, video_output_path, bucket_name)
    video_output_link = objects_detected_data['video_output_link']
    objects_detected_list = objects_detected_data['objects_detected_list']
    json_output_link = objects_detected_data['json_output_link']

    response = {
        'video_output_link': video_output_link,
        'objects_detected': objects_detected_list,
        'json_output_link': json_output_link,
    }
    return jsonify(response)

def upload_chunk_to_gcs(blob, chunk):
    # Upload a chunk to Google Cloud Storage
    blob.upload_from_string(chunk)

@app.route('/upload', methods=['POST'])
def upload_resumable():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        uploaded_file = request.files.get('file')
        if uploaded_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if uploaded_file:
            blob_name = "uploads/" + uploaded_file.filename
            
            client = storage.Client.from_service_account_json('roads-404204.json')
            with GCSObjectStreamUpload(client=client, bucket_name=bucket_name, blob_name=blob_name) as s:
                while True:
                    chunk = uploaded_file.read(s._chunk_size)
                    if not chunk:
                        break
                    s.write(chunk)
                    
            # Set the object's ACL to make it publicly accessible
            s._blob.acl.all().grant_read()
            s._blob.acl.save()

            # Set Content-Disposition header to show the preview in the browser
            s._blob.content_disposition = "inline"
            
            # Set the content type based on the file extension or specify a generic type
            content_type, _ = mimetypes.guess_type(uploaded_file.filename)
            s._blob.content_type = content_type or 'application/octet-stream'
            
            response = {
                'url': s._blob.public_url
            }
            return jsonify(response)

    except Exception as e:
        # Log the exception for debugging
        print(f"Error during file upload: {str(e)}")
        return jsonify({'error': 'File upload failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
    