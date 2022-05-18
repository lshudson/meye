import argparse
import os
import time

os.sys.path += ['expman']
import expman # experiment manager for ml

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import trange

# times how long it takes model to predict pupil center and radius
# should be used if retraining model often - question

def main(args):
    is_run_dir = expman.is_exp_dir(args.model)
    if is_run_dir:
        exp = expman.from_dir(args.model)
        for model_file in ('best_savedmodel', 'best_model.h5', 'last_model.h5'):
            model_path = exp.path_to(model_file)
            if os.path.exists(model_path):
                break
    elif tf.saved_model.contains_saved_model(args.model):
        model_path = args.model
    else:
        print('Cannot find suitable model snapshot in {}'.format(args.model))
        exit(1)

    model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'tf': tf})
    data = np.empty((1, args.rh, args.rw, 1), dtype=np.float32)

    # warm-up
    model.predict(data)

    start = time.time()
    for _ in trange(args.n):
        model.predict(data)
    end = time.time()
    # time it takes model to predict
    elapsed = end - start

    throughput = elapsed / args.n
    fps = args.n / elapsed
    print(f'Total: {elapsed:g}s ({throughput * 1000} ms/img, {fps} fps)')

    timings = pd.Series({'elapsed': elapsed, 'throughput': throughput, 'fps': fps})

    if is_run_dir and not args.output:
        timings_path = exp.path_to('timings.csv')
        timings.to_csv(timings_path)

    if args.output:
        timings.to_csv(args.output)

    return elapsed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict on test video')
    parser.add_argument('model', help='path to model or run dir')
    parser.add_argument('-n', type=int, default=100, help='number of predictions')
    parser.add_argument('-rh', type=int, default=128, help='RoI height (-1 for full height)')
    parser.add_argument('-rw', type=int, default=128, help='RoI width (-1 for full width)')
    parser.add_argument('-o', '--output', help='CSV output file')

    args = parser.parse_args()
    main(args)