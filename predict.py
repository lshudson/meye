import argparse
import time

import imageio
import numpy as np

from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from tqdm import tqdm
from utils import draw_predictions, compute_metrics

# preprocesses image and predicts pupil position and radius as it dilates
# good choice for markers of stimuli from tasks

# begin running time at start of process
start_time = time.time()
def main(args):
    # globals top keep track of pupil coordinate and area across trials
    global new_pupil_y
    global new_pupil_x
    global new_pupil_area

    # initialize data
    pupil_y = 0
    pupil_x = 0
    pupil_area = 0

    # set global values to previously computed values
    new_pupil_y = pupil_y
    new_pupil_x = pupil_x
    new_pupil_area = pupil_area

    video = imageio.get_reader(args.video) # obtain inputted video
    n_frames = video.count_frames() # count number of frames in video
    fps = video.get_meta_data()['fps'] # get frames per second
    frame_w, frame_h = video.get_meta_data()['size'] # get frame width and height

    model = load_model(args.model, compile=False) # load previously saved model
    input_shape = model.input.shape[1:3] # get shape of inputs

    # default RoI (region of interest)
    if None in (args.rl, args.rt, args.rr, args.rb):
        side = min(frame_w, frame_h)
        args.rl = (frame_w - side) / 2 # left
        args.rt = (frame_h - side) / 2 # top
        args.rr = (frame_w + side) / 2 # right
        args.rb = (frame_h + side) / 2 # bottom

    crop = (args.rl, args.rt, args.rr, args.rb)

    def preprocess(frame):
        frame = Image.fromarray(frame)
        eye = frame.crop(crop)
        eye = ImageOps.grayscale(eye) # converts image to grayscale
        eye = eye.resize(input_shape) # resize image to shape of inputs based on model
        return eye

    def predict(eye):
        eye = np.array(eye).astype(np.float32) / 255.0
        eye = eye[None, :, :, None]
        return model.predict(eye) # recursive? calling function inside method

    out_video = imageio.get_writer(args.output_video, fps=fps) # allows one to write data

    frame = np.empty([2,2])
    cropped = map(preprocess(frame), video) # apply preprocess function to all frames of video
    frames_and_predictions = map(lambda x: (x, predict(x)), cropped)

    with open(args.output_csv, 'w') as out_csv:
        print('frame,pupil-area,pupil-x,pupil-y,eye,blink', file=out_csv)
        # enumerate: update counter and value from iterable at same time without making separate variable and having to keep track 
        for idx, (frame, predictions) in enumerate(tqdm(frames_and_predictions, total=n_frames)):
            pupil_map, tags = predictions
            is_eye, is_blink = tags.squeeze()
            # computes coordinate of center of mass and total area
            (pupil_y, pupil_x), pupil_area = compute_metrics(pupil_map, thr=args.thr, nms=True)

            row = [idx, pupil_area, pupil_x, pupil_y, is_eye, is_blink]
            row = ','.join(list(map(str, row)))
            print(row, file=out_csv)

            img = draw_predictions(frame, predictions, thr=args.thr)
            img = np.array(img)
            out_video.append_data(img)

    out_video.close()

    # check if values have changed, if true --> new stimulus
    pupil_y_changed = (new_pupil_y != pupil_y)
    pupil_x_changed = (new_pupil_x != pupil_x)
    pupil_area_changed = (new_pupil_area != pupil_area)

    # if there's a new stimulus, keep note of time
    if pupil_y_changed & pupil_x_changed & pupil_area_changed:
        predict_time = time.time() - start_time
        print(predict_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict on test video')
    parser.add_argument('model', type=str, help='Path to model')
    parser.add_argument('video', type=str, default='<video0>', help='Video file to process (use \'<video0>\' for webcam)')

    parser.add_argument('-t', '--thr', type=float, default=0.5, help='Map Threshold')
    parser.add_argument('-rl', type=int, help='RoI X coordinate of top left corner')
    parser.add_argument('-rt', type=int, help='RoI Y coordinate of top left corner')
    parser.add_argument('-rr', type=int, help='RoI X coordinate of right bottom corner')
    parser.add_argument('-rb', type=int, help='RoI Y coordinate of right bottom corner')

    parser.add_argument('-ov', '--output-video', default='predictions.mp4', help='Output video')
    parser.add_argument('-oc', '--output-csv', default='pupillometry.csv', help='Output CSV')

    args = parser.parse_args()
    main(args)  