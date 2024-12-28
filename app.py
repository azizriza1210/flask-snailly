from flask import Flask, request, jsonify
import subprocess
import requests
from PIL import Image
from io import BytesIO
import json
import predict_image
import predict_video
import httpx
import base64
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import urllib.request
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pytube import YouTube
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

app = Flask(__name__)
CORS(app)

model = YOLO("/home/snailly/mysite/flask-snailly/best.pt")


# IMAGE FUNCTION
# VIDEO DETECTION
def get_video_links(url):
    """
    Mendapatkan semua link video dari halaman web
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Periksa apakah permintaan berhasil
        soup = BeautifulSoup(response.content, 'html.parser')

        # Cari semua tag <video> atau link dengan ekstensi video
        video_links = []

        # Cari dari tag <video> -> <source>
        for video_tag in soup.find_all('video'):
            for source in video_tag.find_all('source'):
                video_src = source.get('src')
                if video_src:
                    video_links.append(urljoin(url, video_src))  # URL absolut

        # Tambahkan link langsung ke file video (misalnya .mp4, .webm)
        for link in soup.find_all('a', href=True):
            if link['href'].endswith(('mp4', 'webm', 'avi', 'mov')):
                video_links.append(urljoin(url, link['href']))  # URL absolut

        return list(set(video_links))  # Hilangkan duplikat

    except Exception as e:
        print(f"Error occurred while fetching video links: {e}")
        return []

def print_progress(iteration, total, prefix='', suffix='', decimals=3, bar_length=100):
    """
    Call in a loop to create standard out progress bar
    :param iteration: current iteration
    :param total: total iterations
    :param prefix: prefix string
    :param suffix: suffix string
    :param decimals: positive number of decimals in percent complete
    :param bar_length: character length of bar
    :return: None
    """

    format_str = "{0:." + str(decimals) + "f}"  # format the % done number string
    percents = format_str.format(100 * (iteration / float(total)))  # calculate the % done
    filled_length = int(round(bar_length * iteration / float(total)))  # calculate the filled bar length
    bar = '#' * filled_length + '-' * (bar_length - filled_length)  # generate the bar string
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),  # write out the bar
    sys.stdout.flush()  # flush to stdout


def extract_frames(video_path, frames_dir, overwrite=False, start=-1, end=-1, every=1):
    """
    Extract frames from a video using OpenCVs VideoCapture
    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :return: count of images saved
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    assert os.path.exists(video_path)  # assert the video file exists

    capture = cv2.VideoCapture(video_path)  # open the video using OpenCV

    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.set(1, start)  # set the starting frame of the capture
    frame = start  # keep track of which frame we are up to, starting from start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    saved_count = 0  # a count of how many frames we have saved

    while frame < end:  # lets loop through the frames until the end

        _, image = capture.read()  # read an image from the capture

        if while_safety > 500:  # break the while if our safety maxs out at 500
            break

        # sometimes OpenCV reads None's during a video, in which case we want to just skip
        if image is None:  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            continue  # skip

        if frame % every == 0:  # if this is a frame we want to write out based on the 'every' argument
            while_safety = 0  # reset the safety count
            save_path = os.path.join(frames_dir, video_filename, "{:010d}.jpg".format(frame))  # create the save path
            if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
                cv2.imwrite(save_path, image)  # save the extracted image
                saved_count += 1  # increment our counter by one

        frame += 1  # increment our frame count

    capture.release()  # after the while has finished close the capture

    return saved_count  # and return the count of the images we saved


def video_to_frames(video_path, frames_dir, overwrite=False, every=10, chunk_size=1000):
    """
    Extracts the frames from a video using multiprocessing
    :param video_path: path to the video
    :param frames_dir: directory to save the frames
    :param overwrite: overwrite frames if they exist?
    :param every: extract every this many frames
    :param chunk_size: how many frames to split into chunks (one chunk per cpu core process)
    :return: path to the directory where the frames were saved, or None if fails
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    # make directory to save frames, its a sub dir in the frames_dir with the video name
    os.makedirs(os.path.join(frames_dir, video_filename), exist_ok=True)

    capture = cv2.VideoCapture(video_path)  # load the video
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # get its total frame count
    capture.release()  # release the capture straight away

    if total < 1:  # if video has no frames, might be and opencv error
        print("Video has no frames. Check your OpenCV + ffmpeg installation")
        return None  # return None

    chunk_size = min(chunk_size,total-1)
    frame_chunks = [[i, i+chunk_size] for i in range(0, total, chunk_size)]  # split the frames into chunk lists
    frame_chunks[-1][-1] = min(frame_chunks[-1][-1], total-1)  # make sure last chunk has correct end frame, also handles case chunk_size < total

    prefix_str = "Extracting frames from {}".format(video_filename)  # a prefix string to be printed in progress bar

    # execute across multiple cpu cores to speed up processing, get the count automatically
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:

        futures = [executor.submit(extract_frames, video_path, frames_dir, overwrite, f[0], f[1], every)
                   for f in frame_chunks]  # submit the processes: extract_frames(...)

        for i, f in enumerate(as_completed(futures)):  # as each process completes
            print_progress(i, len(frame_chunks)-1, prefix=prefix_str, suffix='Complete')  # print it's progress

    return os.path.join(frames_dir, video_filename)  # when done return the directory containing the frames# END VIDEO DETECTION


@app.route('/')
def home():
    halo_web = "HALLLOOOO!!!!"
    return halo_web

@app.route('/image-prediction',methods=['POST'])
def image_prediction():
    file = request.files.get('file')
    if not file:
        return json.dumps({'error': 'No file uploaded'}), 400

    filepath = save_image(file)
    results = predict_image(filepath)
    class_name = get_label(results)

    return class_name

# Route untuk menerima data via POST
@app.route('/predict-image', methods=['POST'])
def predict():
    try:
        # Mendapatkan URL gambar dari request
        data = request.json
        image_url = data.get('image_url')

        if not image_url:
            return jsonify({"error": "No image URL provided"}), 400


        # Nama file untuk menyimpan gambar yang diunduh
        filename = "downloaded_image.jpg"

        # Mendownload gambar dari URL
        urllib.request.urlretrieve(image_url, filename)

        # Buka file gambar menggunakan PIL
        try:
            img = Image.open(filename)  # Membuka gambar yang diunduh
        except Exception as e:
            return jsonify({"error": f"Failed to open image: {str(e)}"}), 500

        try:
            # Simpan gambar sementara di server
            img_path = "temp_image.jpg"
            img.save(img_path)  # Menyimpan gambar sebagai file sementara
            print(f"Gambar sementara disimpan di: {img_path}")
            prediction = predict_image.predict_image(filename)

            print(json.dumps({"prediction": prediction}))
        #     prediction_output = json.loads(result.stdout)  # Parsing output JSON dari subprocess
        except json.JSONDecodeError:
            return jsonify({"error": "Failed to parse prediction result"}), 500

        # Kembalikan hasil prediksi
        return jsonify({"hasil": prediction})

    except Exception as e:
        print("INI ERROR : ",e)
        return jsonify({"error": str(e)}), 500

@app.route('/video-prediction',methods=['POST'])
def video_prediction():
    url = request.files.get('url')
    name = "video.mp4"
    try:
        print("Downloading starts...\n")
        download_video(url, name)
        print("Download completed..!!")

        video_to_frames(video_path=name, frames_dir='D:/Codelabs/Snailly-Video/test_frames', overwrite=False, every=20, chunk_size=500)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    app.run(debug=True)
