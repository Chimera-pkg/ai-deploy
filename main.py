import os
import flask
from flask import Flask, request, jsonify
from helpers import load_model, process_image, save_summary_image, process_video, summary_video, save_summary_video, upload_to_cloud_storage
from urllib.parse import quote
import fileapp
from google.cloud import storage
import os
import io

os.environ['GOOGLE_CLOUD_PROJECT'] = 'roads-404204'
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3600  # Set the timeout to 1 hour (in seconds)
app.config['MAX_CONTENT_LENGTH'] = 2147483648   # 2GB limit
bucket_name = 'asia.artifacts.roads-404204.appspot.com'

@app.route('/detect_image/<image_filename>', methods=['GET', 'POST'])
def detect_image(image_filename):
    # Define the path for saving the output image and JSON
    base_path, _ = os.path.splitext(image_filename)
    image_output_path = f'image-detection/{base_path}_output{_}'
    json_output_path = f'image-detection/{base_path}_summary.json'

    try:
        model = load_model()
        objects_detected_data = process_image(model, image_filename, image_output_path)
        image_output_link = objects_detected_data['image_output_link']
        objects_detected = objects_detected_data['objects_detected']
        save_summary_image(objects_detected, json_output_path)

        # Upload json_output to Google Cloud Storage
        json_output_link = upload_to_cloud_storage(json_output_path, bucket_name)

        response = {
            'image_output_link': image_output_link,
            'objects_detected': objects_detected,
            'json_output_link': json_output_link,
        }
        return jsonify(response)

    except Exception as e:
        # Log the exception for debugging
        print(f"Error during road damage detection: {str(e)}")
        return jsonify({'error': 'Road damage detection failed'}), 500

@app.route('/detect_video/<video_filename>', methods=['GET', 'POST'])
def detect_video(video_filename):
    # Define the path for saving the annotated video and JSON
    base_path, _ = os.path.splitext(video_filename)
    output_suffix = '_output'
    summary_suffix = '_summary'
    video_output_path = f'video-detection/{base_path}{output_suffix}{_}'
    json_output_path = f'video-detection/{base_path}{summary_suffix}.json'

    try:
        model = load_model()
        objects_detected_data = process_video(model, video_filename, video_output_path, bucket_name)
        video_output_link = objects_detected_data['image_output_link']
        objects_detected_list = objects_detected_data['objects_detected_list']
        summary_objects_detected_list = summary_video(objects_detected_list)
        save_summary_video(json_output_path, summary_objects_detected_list)

        # Upload json_output to Google Cloud Storage
        json_output_link = upload_to_cloud_storage(json_output_path, bucket_name)

        response = {
            'video_output_link': video_output_link,
            'objects_detected': objects_detected_list,
            'json_output_link': json_output_link,
        }
        return jsonify(response)

    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 400

def upload_chunk_to_gcs(blob, chunk):
    # Upload a chunk to Google Cloud Storage
    blob.upload_from_string(chunk)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        uploaded_file = request.files.get('file')
        if uploaded_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if uploaded_file:
            # Convert the filename to a URL-friendly format
            filename = quote(uploaded_file.filename)
            
            gcs_client = storage.Client.from_service_account_json('roads-404204.json')
            storage_bucket = gcs_client.get_bucket(bucket_name)
            blob = storage_bucket.blob(uploaded_file.filename)

            chunk_size = 30 * 1024 * 1024  # 10 MB (adjust as needed)
            chunk = uploaded_file.read(chunk_size)

            while chunk:
                upload_chunk_to_gcs(blob, chunk)
                chunk = uploaded_file.read(chunk_size)

            # Set the object's ACL to make it publicly accessible
            blob.acl.all().grant_read()
            blob.acl.save()

            response = {
                'url': blob.public_url
            }
            return jsonify(response)

    except Exception as e:
        # Log the exception for debugging
        print(f"Error during file upload: {str(e)}")
        return jsonify({'error': 'File upload failed'}), 500
   
@app.route('/upload-chunk', methods=['POST'])
def upload_chunk():
    if 'file' in request.files:
        chunk = request.files['file'].read()
        filename = request.files['file'].filename
        save_chunk(chunk, filename)
        return 'Chunk uploaded successfully', 200
    else:
        return 'No file part in the request', 400

def save_chunk(chunk, filename):
    if 'file_data' not in save_chunk.__dict__:
        save_chunk.file_data = io.BytesIO()
    save_chunk.file_data.write(chunk)

    save_chunk.file_data.seek(0)
    upload_to_gcs(save_chunk.file_data, filename)

def upload_to_gcs(file_data, filename):
    client = storage.Client.from_service_account_json('roads-404204.json')
    CHUNK_SIZE = 1024 * 1024 * 30
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(f'uploads/{filename}', chunk_size=CHUNK_SIZE)
    blob.upload_from_file(file_data, content_type='application/octet-stream')
 
if __name__ == '__main__':
    app.run(debug=True)
    