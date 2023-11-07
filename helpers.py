import os
import cv2
import json
from PIL import Image
from ultralytics import YOLO
import tempfile
from google.cloud import storage
from google.cloud.storage.blob import Blob  # Corrected import
import os
from io import BytesIO
import io

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
    try:
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
    
    finally:
        # Clean up the temporary directory
        if os.path.exists(tmp_dir):
            for file_name in os.listdir(tmp_dir):
                file_path = os.path.join(tmp_dir, file_name)
                os.remove(file_path)
            os.rmdir(tmp_dir)

def process_video(model, video_filename, video_output_path, bucket_name):
    # Create a temporary directory for the video
    tmp_dir = tempfile.mkdtemp()
    
    # Create a subdirectory named "video-detection" within tmp_dir
    video_detection_dir = os.path.join(tmp_dir, "video-detection")
    os.makedirs(video_detection_dir, exist_ok=True)

    try:
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
                
        # Compress the video using FFmpeg (imageio-ffmpeg library)
        compressed_video_path = os.path.join(tmp_dir, f"compressed_{local_video_output_path}")
        os.system(f"ffmpeg -i {local_video_output_path} -c:v libx264 -crf 23 -preset medium {compressed_video_path}")

        # Upload the annotated video to Google Cloud Storage
        client = storage.Client.from_service_account_json('roads-404204.json')
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"uploads/{compressed_video_path}")

        # Upload the annotated image
        blob.upload_from_filename(compressed_video_path)

        # Make the object public
        blob.acl.all().grant_read()
        blob.acl.save()
        
        video_output_link = blob.public_url
        
        return {
            'objects_detected_list': objects_detected_list,
            'video_output_link': video_output_link,
            'json_output_link': json_blob.public_url
        }
        
    finally:
        # Clean up the temporary directory
        if os.path.exists(tmp_dir):
            for file_name in os.listdir(tmp_dir):
                file_path = os.path.join(tmp_dir, file_name)
                os.remove(file_path)
            os.rmdir(tmp_dir)

def save_summary_video(json_output_path, objects_detected_list):
    with open(json_output_path, 'w') as json_file:
        json.dump(objects_detected_list, json_file, indent=4)