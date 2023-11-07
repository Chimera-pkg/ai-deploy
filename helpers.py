import os
import cv2
import json
from PIL import Image
from ultralytics import YOLO
import tempfile
from google.cloud import storage
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
 
def save_summary_image(objects_detected, json_output_path):
    # Create a temporary directory for the image
    tmp_dir = tempfile.mkdtemp()
    
    # Ensure that the directory where the JSON file will be saved exists
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

    # Create the full path to the JSON file in the temporary directory
    json_output_path = os.path.join(tmp_dir, json_output_path)
    with open(json_output_path, 'w') as json_file:
        json.dump(objects_detected, json_file, indent=4)

# def process_video(model, video_filename, video_output_path, bucket_name):
#     # Download the video from Google Cloud Storage
#     download_from_cloud_storage(video_filename, video_output_path, bucket_name)

#     cap = cv2.VideoCapture(video_filename)
#     success, frame = cap.read()
#     H, W, _ = frame.shape
#     out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'XVID'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

#     objects_detected_list = []
#     last_objects_detected = {}
#     frame_count = 0

#     while cap is not None and cap.isOpened():
#         success, frame = cap.read()
#         if success:
#             frame_count += 1
#             results = model.predict(frame)
#             annotated_frame = results[0].plot()
#             out.write(annotated_frame)

#             # Summary
#             data_json = json.loads(results[0].tojson())
#             objects_detected = {}
#             for data in data_json:
#                 class_name = data["name"]
#                 confidence = round(data["confidence"], 2)

#                 if class_name in objects_detected:
#                     objects_detected[class_name]["confidence_sum"] += confidence
#                     objects_detected[class_name]["count"] += 1
#                 else:
#                     objects_detected[class_name] = {
#                         "confidence_sum": round(confidence, 2),
#                         "count": 1
#                     }

#             for class_name, info in objects_detected.items():
#                 average_confidence = info["confidence_sum"] / info["count"]
#                 info["average_confidence"] = round(average_confidence * 100, 2)

#             # Check object detected
#             unique_objects_detected = {}
#             for class_name, info in objects_detected.items():
#                 if class_name not in last_objects_detected or frame_count - last_objects_detected[class_name] > 1:
#                     unique_objects_detected[class_name] = info

#             last_objects_detected = {class_name: frame_count for class_name in objects_detected}

#             if unique_objects_detected:
#                 objects_detected_list.append(unique_objects_detected)

#         else:
#             break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

#     # Upload the annotated video to Google Cloud Storage
#     video_output_link = upload_to_cloud_storage(video_output_path, bucket_name)

#     return {
#         'objects_detected_list': objects_detected_list,
#         'video_output_link': video_output_link
#     }

def process_video(model, video_filename, video_output_path, bucket_name):
    # Download the video from Google Cloud Storage
    download_from_cloud_storage(video_filename, video_output_path, bucket_name)

    cap = cv2.VideoCapture(video_output_path)
    success, frame = cap.read()
    H, W, _ = frame.shape

    # Initialize the Google Cloud Storage client
    storage_client = storage.Client.from_service_account_json('roads-404204.json')

    # Specify the bucket where your videos are stored
    bucket = storage_client.bucket(bucket_name)

    out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'XVID'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    objects_detected_list = []
    last_objects_detected = {}
    frame_count = 0

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

    # Create a temporary directory for video output
    tmp_dir = tempfile.mkdtemp()

    # Create a local path for video output
    local_video_output_path = os.path.join(tmp_dir, video_output_path)

    # Upload the annotated video to Google Cloud Storage
    video_output_link = upload_to_cloud_storage(local_video_output_path, bucket_name)

    # Serialize objects_detected_list to JSON
    objects_detected_json = json.dumps(objects_detected_list, indent=4)

    # Define the file name for the JSON file
    json_output_filename = f"video-detection/{video_output_path}_summary.json"

    # Create a GCS blob and upload the JSON file directly
    json_blob = bucket.blob("uploads/" + json_output_filename)
    json_blob.upload_from_string(objects_detected_json, content_type='application/json')

    return {
        'objects_detected_list': objects_detected_list,
        'video_output_link': video_output_link,
        'json_output_link': json_blob.public_url
    }

def summary_video(objects_detected_list):
    merged_objects_detected_list = {}
    for item in objects_detected_list:
        for key, value in item.items():
            if key in merged_objects_detected_list:
                merged_objects_detected_list[key]["confidence_sum"] += value["confidence_sum"]
                merged_objects_detected_list[key]["count"] += value["count"]
            else:
                merged_objects_detected_list[key] = value

    for key, value in merged_objects_detected_list.items():
        confidence_sum = value["confidence_sum"]
        count = value["count"]
        average_confidence = (confidence_sum / count) * 100
        value["average_confidence"] = f"{average_confidence:.2f}%"

    summary_objects_detected_list = [{key: value} for key, value in merged_objects_detected_list.items()]
    return summary_objects_detected_list

def save_summary_video(json_output_path, objects_detected_list):
    with open(json_output_path, 'w') as json_file:
        json.dump(objects_detected_list, json_file, indent=4)

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