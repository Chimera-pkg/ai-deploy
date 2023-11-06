import os
from flask import Flask, request, jsonify
from helpers import load_model, process_image, save_summary_image, process_video, summary_video, save_summary_video

app = Flask(__name__)

@app.route('/detect_image/<image_path>', methods=['GET', 'POST'])
def detect_image(image_path):
    base_path, extension = os.path.splitext(image_path)
    image_output_path = f'{base_path}_output{extension}'
    json_output_path = f'{base_path}_summary.json'
    
    try:
        model = load_model()
        objects_detected = process_image(model, image_path, image_output_path)
        save_summary_image(objects_detected, json_output_path)
        response = {
            'image_output_path': image_output_path,
            'objects_detected': objects_detected,
            'json_output_path': json_output_path,
        }
        return jsonify(response)

    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 400


@app.route('/detect_video/<video_path>', methods=['GET', 'POST'])
def detect_video(video_path):
    base_path, extension = os.path.splitext(video_path)
    output_suffix = '_output'
    summary_suffix = '_summary'
    video_output_path = f'{base_path}{output_suffix}{extension}'
    json_output_path = f'{base_path}{summary_suffix}.json'

    try:
        model = load_model()
        objects_detected_list = process_video(model, video_path, video_output_path)
        summary_objects_detected_list = summary_video(objects_detected_list)
        save_summary_video(json_output_path, summary_objects_detected_list)
        response = {
            'video_output_path': video_output_path,
            'objects_detected': objects_detected_list,
            'json_output_path': json_output_path,
        }
        return jsonify(response)

    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 400

if __name__ == '__main__':
    app.run(debug=True)
