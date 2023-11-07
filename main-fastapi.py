import os
from fastapi import FastAPI, UploadFile, File
from google.cloud import storage
from urllib.parse import quote
from helpers import load_model, process_image, process_video
import io

os.environ['GOOGLE_CLOUD_PROJECT'] = 'roads-404204'
app = FastAPI()

bucket_name = 'asia.artifacts.roads-404204.appspot.com'

@app.post('/detect_image/{image_filename}')
async def detect_image(image_filename: str, file: UploadFile):
    base_path, _ = os.path.splitext(image_filename)
    image_output_path = os.path.join('image-detection', f'{base_path}_output{_}')

    model = load_model()
    objects_detected_data = await process_image(model, file, image_filename, image_output_path)
    return objects_detected_data

@app.post('/detect_video/{video_filename}')
async def detect_video(video_filename: str, file: UploadFile):
    base_path, _ = os.path.splitext(video_filename)
    output_suffix = '_output'

    video_output_path = os.path.join('video-detection', f'{base_path}{output_suffix}{_}')

    model = load_model()
    objects_detected_data = await process_video(model, file, video_filename, video_output_path, bucket_name)
    return objects_detected_data

def upload_chunk_to_gcs(blob, chunk):
    # Upload a chunk to Google Cloud Storage
    blob.upload_from_string(chunk)

@app.post('/upload')
async def upload(file: UploadFile):
    try:
        # Convert the filename to a URL-friendly format
        filename = quote(file.filename)

        gcs_client = storage.Client.from_service_account_json('roads-404204.json')
        storage_bucket = gcs_client.get_bucket(bucket_name)
        blob = storage_bucket.blob("uploads/" + filename)

        chunk_size = 30 * 1024 * 1024  # 10 MB (adjust as needed)
        chunk = await file.read(chunk_size)

        while chunk:
            upload_chunk_to_gcs(blob, chunk)
            chunk = await file.read(chunk_size)

        # Set the object's ACL to make it publicly accessible
        blob.acl.all().grant_read()
        blob.acl.save()

        response = {
            'url': blob.public_url
        }
        return response

    except Exception as e:
        # Log the exception for debugging
        print(f"Error during file upload: {str(e)}")
        return {'error': 'File upload failed'}, 500

if __name__ == '__main__':
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
