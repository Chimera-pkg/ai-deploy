import os
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from helpers import load_model, process_image, process_video
from urllib.parse import quote
from google.cloud import storage

os.environ['GOOGLE_CLOUD_PROJECT'] = 'roads-404204'

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

bucket_name = 'asia.artifacts.roads-404204.appspot.com'

@app.post("/detect_image/{image_filename}")
async def detect_image(image_filename: str):
    base_path, _ = os.path.splitext(image_filename)
    image_output_path = os.path.join('image-detection', f'{base_path}_output{_}')

    model = load_model()
    objects_detected_data = await process_image(model, image_filename, image_output_path)
    image_output_link = objects_detected_data['image_output_link']
    objects_detected = objects_detected_data['objects_detected']
    json_output_link = objects_detected_data['json_output_link']

    response = {
        'image_output_link': image_output_link,
        'objects_detected': objects_detected,
        'json_output_link': json_output_link,
    }
    return response

@app.post("/detect_video/{video_filename}")
async def detect_video(video_filename: str):
    base_path, _ = os.path.splitext(video_filename)
    output_suffix = '_output'

    video_output_path = os.path.join('video-detection', f'{base_path}{output_suffix}{_}')

    model = load_model()
    objects_detected_data = await process_video(model, video_filename, video_output_path, bucket_name)
    video_output_link = objects_detected_data['video_output_link']
    objects_detected_list = objects_detected_data['objects_detected_list']
    json_output_link = objects_detected_data['json_output_link']

    response = {
        'video_output_link': video_output_link,
        'objects_detected': objects_detected_list,
        'json_output_link': json_output_link,
    }
    return response

@app.post("/upload")
async def upload(file: UploadFile):
    if not file:
        return {'error': 'No file part'}, 400

    filename = quote(file.filename)
    gcs_client = storage.Client.from_service_account_json('roads-404204.json')
    storage_bucket = gcs_client.get_bucket(bucket_name)
    blob = storage_bucket.blob(f"uploads/{file.filename}")

    tmp_file = f'/tmp/{file.filename}'
    file.save(tmp_file)
    
    storage_client: storage.Client = storage.Client.from_service_account_json('roads-404204.json')
    bucket: storage.Bucket = storage_client.bucket(bucket_name)
    blob: storage.Blob = bucket.blob(f"uploads/{file.filename}")
    blob.upload_from_filename(tmp_file)
    blob.make_public()
    os.remove(tmp_file)

    response = {
        'url': blob.public_url
    }
    return response

def upload_chunk_to_gcs(blob, chunk):
    blob.upload_from_string(chunk)
