import os
import cv2
import json
from PIL import Image
from ultralytics import YOLO
import tempfile
from google.cloud import storage
from google.cloud.storage.blob import Blob  # Corrected import
from google.auth.transport.requests import AuthorizedSession
from google.resumable_media import requests, common
import os
import io

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

def load_model():
    model_path = os.path.join('models', 'best_rdd_final.pt')
    return YOLO(model_path)

def download_from_cloud_storage(remote_file_name, local_file_name, bucket_name):
    client = storage.Client.from_service_account_json('roads-404204.json')
    bucket = client.bucket(bucket_name)
    blob = bucket.blob("uploads/" + remote_file_name)
    blob.download_to_filename(local_file_name)

def process_image(model, image_filename, image_output_filename):
    # Replace 'your-bucket-name' with your actual Google Cloud Storage bucket name
    bucket_name = 'asia.artifacts.roads-404204.appspot.com'

    # Create a temporary directory for the image
    tmp_dir = tempfile.mkdtemp()
    # try:
    # Define the local file path for the downloaded image
    local_image_path = os.path.join(tmp_dir, image_filename)

    # Download the image from Google Cloud Storage to the temporary directory
    download_from_cloud_storage(image_filename, local_image_path, bucket_name)

    # Initialize the Google Cloud Storage client
    storage_client = storage.Client.from_service_account_json('roads-404204.json')

    # Specify the bucket where your images are stored
    bucket = storage_client.bucket(bucket_name)

    # Check if the image exists in the bucket
    blob = bucket.blob("uploads/" + image_filename)

    if not blob.exists():
        return {'error': 'File not found in the bucket'}, 404

    # Load the image from the temporary directory
    with Image.open(local_image_path) as image:
        # Perform YOLO model prediction
        results = model.predict(image)

    # Annotate the image
    annotated_img = results[0].plot()
    annotated_img = Image.fromarray(annotated_img[..., ::-1])

    # Create a BytesIO buffer to store the annotated image
    annotated_image_buffer = io.BytesIO()

    # Save the annotated image to the buffer in JPEG format
    annotated_img.save(annotated_image_buffer, format='JPEG')

    # Create a GCS blob and upload the annotated image directly
    blob = bucket.blob("uploads/" + image_output_filename)
    blob.upload_from_string(annotated_image_buffer.getvalue(), content_type='image/jpeg')

    # Set the object's ACL to make it publicly accessible
    blob.acl.all().grant_read()
    blob.acl.save()

    data_json = json.loads(results[0].tojson())
    objects_detected = {}

    for data in data_json:
        class_name = data["name"]
        confidence = round(data["confidence"], 2)

        if class_name in objects_detected:
            objects_detected[class_name]["confidence_sum"] += confidence
            objects_detected[class_name]["count"] += 1
        else:
            objects_detected[class_name] = {
                "confidence_sum": round(confidence, 2),
                "count": 1
            }

    for class_name, info in objects_detected.items():
        average_confidence = info["confidence_sum"] / info["count"]
        objects_detected[class_name]["average_confidence"] = round(average_confidence, 2)

    # Serialize objects_detected to JSON
    objects_detected_json = json.dumps(objects_detected, indent=4)

    # Define the file name for the JSON file
    json_output_filename = f"image-detection/{image_output_filename}_summary.json"

    # Create a GCS blob and upload the JSON file directly
    json_blob = bucket.blob("uploads/" + json_output_filename)
    json_blob.upload_from_string(objects_detected_json, content_type='application/json')

    # Set the object's ACL to make it publicly accessible
    json_blob.acl.all().grant_read()
    json_blob.acl.save()

    return {
        'objects_detected': objects_detected,
        'image_output_link': blob.public_url,
        'json_output_link': json_blob.public_url
    }
    
    # finally:
    #     # Clean up the temporary directory
    #     if os.path.exists(tmp_dir):
    #         for file_name in os.listdir(tmp_dir):
    #             file_path = os.path.join(tmp_dir, file_name)
    #             os.remove(file_path)
    #         os.rmdir(tmp_dir)

def upload_to_cloud_storage(local_path, bucket_name, cloud_path):
    """Upload a file to Google Cloud Storage.

    Args:
        local_path (str): The local file path to be uploaded.
        bucket_name (str): The name of the Google Cloud Storage bucket.
        cloud_path (str): The destination path in the cloud storage bucket.

    Returns:
        str: The public URL of the uploaded file.
    """
    # Initialize the Google Cloud Storage client
    storage_client = storage.Client.from_service_account_json('roads-404204.json')

    # Specify the bucket where the file will be stored
    bucket = storage_client.bucket(bucket_name)

    # Create a GCS blob and upload the file
    blob = Blob(cloud_path, bucket)
    with open(local_path, "rb") as file:
        blob.upload_from_file(file, content_type='application/octet-stream')

    # Set the object's ACL to make it publicly accessible
    blob.acl.all(). grant_read()
    blob.acl.save()

    # Return the public URL of the uploaded file
    return blob.public_url

def process_video(model, video_filename, video_output_path, bucket_name):
    # Create a temporary directory for the video
    tmp_dir = tempfile.mkdtemp()
    
    # Create a subdirectory named "video-detection" within tmp_dir
    video_detection_dir = os.path.join(tmp_dir, "video-detection")
    os.makedirs(video_detection_dir, exist_ok=True)

    # try:
    # Define the local file path for the downloaded video
    local_video_path = os.path.join(tmp_dir, video_filename)

    # Download the video from Google Cloud Storage
    download_from_cloud_storage(video_filename, local_video_path, bucket_name)

    cap = cv2.VideoCapture(local_video_path)
    success, frame = cap.read()
    H, W, _ = frame.shape

    objects_detected_list = []
    last_objects_detected = {}
    frame_count = 0

    # Use the same temporary directory for video output
    local_video_output_path = os.path.join(tmp_dir, os.path.join(tmp_dir, video_output_path.format(frame_index=frame_count)))

    # Define the VideoWriter 'out' here
    out = cv2.VideoWriter(local_video_output_path, cv2.VideoWriter_fourcc(*'XVID'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    while cap is not None and cap.isOpened():
        success, frame = cap.read()
        if success:
            frame_count += 1
            results = model.predict(frame)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

            # Summary
            data_json = json.loads(results[0].tojson())
            objects_detected = {}
            for data in data_json:
                class_name = data["name"]
                confidence = round(data["confidence"], 2)

                if class_name in objects_detected:
                    objects_detected[class_name]["confidence_sum"] += confidence
                    objects_detected[class_name]["count"] += 1
                else:
                    objects_detected[class_name] = {
                        "confidence_sum": round(confidence, 2),
                        "count": 1
                    }

            for class_name, info in objects_detected.items():
                average_confidence = info["confidence_sum"] / info["count"]
                info["average_confidence"] = round(average_confidence * 100, 2)

            # Check object detected
            unique_objects_detected = {}
            for class_name, info in objects_detected.items():
                if class_name not in last_objects_detected or frame_count - last_objects_detected[class_name] > 1:
                    unique_objects_detected[class_name] = info

            last_objects_detected = {class_name: frame_count for class_name in objects_detected}

            if unique_objects_detected:
                objects_detected_list.append(unique_objects_detected)

        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()  

    # Initialize the Google Cloud Storage client
    storage_client = storage.Client.from_service_account_json('roads-404204.json')

    # Specify the bucket where your videos are stored
    bucket = storage_client.bucket(bucket_name)
    
    # Serialize objects_detected_list to JSON
    objects_detected_json = json.dumps(objects_detected_list, indent=4)

    # Define the file name for the JSON file
    json_output_filename = f"video-detection/{video_filename}_summary.json"

    # Upload the summary JSON to Google Cloud Storage
    # Create a GCS blob and upload the JSON file directly
    json_blob = bucket.blob("uploads/" + json_output_filename)
    json_blob.upload_from_string(objects_detected_json, content_type='application/json')

    # Set the object's ACL to make it publicly accessible
    json_blob.acl.all().grant_read()
    json_blob.acl.save()

    # Upload the annotated video to Google Cloud Storage
    video_output_link = upload_to_cloud_storage(local_video_output_path, bucket_name)

    return {
        'objects_detected_list': objects_detected_list,
        'video_output_link': video_output_link,
        'json_output_link': json_blob.public_url
    }
    # finally:
    #     # Clean up the temporary directory
    #     if os.path.exists(tmp_dir):
    #         for file_name in os.listdir(tmp_dir):
    #             file_path = os.path.join(tmp_dir, file_name)
    #             os.remove(file_path)
    #         os.rmdir(tmp_dir)

def upload_to_cloud_storage(image_output_filename, bucket_name):
    client = storage.Client.from_service_account_json('roads-404204.json')
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"uploads/{image_output_filename}")

    # Upload the annotated image
    blob.upload_from_filename(image_output_filename)

    # Make the object public
    blob.acl.all().grant_read()
    blob.acl.save()

    # Return the public URL of the uploaded image
    return blob.public_url