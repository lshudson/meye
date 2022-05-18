import numpy as np
import matplotlib.pyplot as plt
import time

from PIL import Image, ImageDraw
from scipy.ndimage import center_of_mass, label, sum as area

# computes center and area of eye, draws predictions on data, and visualizes eye (no ml)
# since no ml computations should not take too much time 
# two functions in this file are called in predict.py - only need predict.py

# compute and return area for largest connected component
def nms_on_area(self, x, s):  
    # x is a binary image (original matrix of 0s and 1s), s is a structuring element (new matrix of 0s and 1s 
    # used to probe an image for a region of interest; the center pixel identifies the pixel being processed)
    labels, num_labels = label(self.x, structure=self.s)  # find connected components
    if num_labels > 1:
        indexes = np.arange(1, num_labels + 1) # start from 1 so last index needs to be incremented by 1
        areas = area(self.x, labels, indexes)  # compute area for each connected components
        
        biggest = max(zip(areas, indexes))[1]  # get index of largest component
        self.x[labels != biggest] = 0  # discard other components
    return self.x

# compute center and total area
def compute_metrics(self, p, thr=None, nms=False):
    self.p = self.p.squeeze() # removes all dimensions of size 1

    if thr:
        self.p = self.p > thr
        # remove redundant bounding boxes in object detection
        if nms:  # perform non-maximum suppression: keep only largest area
            self.s = np.ones((3, 3))  # connectivity structure
            self.p = nms_on_area(self.p, self.s)

    center = center_of_mass(self.p)
    area = self.p.sum()
    # returns center of mass and area of pupil
    return center, area

# makes eye visualizable for future analysis
def visualizable(self, x, y, alpha=(.5, .5), thr=0):
    # converts string to title cased version
    xx = np.tile(x, (3,))  # Gray -> RGB: repeat channels 3 times
    yy = (self.y, ) + (np.zeros_like(self.x),) * (3 - self.y.shape[-1])
    yy = np.concatenate(yy, axis=-1)  # add a zero channels to pad to RGB
    mask = yy.max(axis=-1, keepdims=True) > thr  # blend only where a prediction is present, if greater than threshold
    # mask = mask[:, :, None]
    # returns indices of elements in input array where condition is satisfied
    return np.where(mask, alpha[0] * xx + alpha[1] * yy, xx)

# draw predictions onto image of eye
def draw_predictions(self, image, predictions, thr=None):
    # converted copy of image
    # rgb color model with fourth alpha channel that indicates how opaque each pixel is 
    x = self.image.convert('RGBA')

    maps, tags = self.predictions
    maps = maps[0] if maps.ndim == 4 else maps
    eye, blink = tags.squeeze()
    alpha = maps.max(axis=-1, keepdims=True)
    alpha = alpha > thr if thr is not None else alpha

    n_pad = 3 - maps.shape[-1]
    zero_channels = np.zeros(image.size + (n_pad,))
    y = np.concatenate((maps, zero_channels, alpha), axis=-1)  # add pad and masked alpha channel
    y = (y * 255).astype(np.uint8)
    y = Image.fromarray(y).convert('RGBA')

    # combining one image with a background to create the appearance of partial or full transparency
    preview = Image.alpha_composite(x, y)
    draw = ImageDraw.Draw(preview)
    draw.text((5, 5), 'E: {: >3.1%}  B:{: >3.1%}'.format(eye, blink), fill=(0, 0, 255))
    # draw.text((5, image.height - 5), ''.format(blink), fill=(255, 0, 0))

    # returns alpha composite image
    return preview

# graphs grid of eye
def visualize(self, x, y, out=None, thr=0, n_cols=4, width=20):
    n_rows = len(self.x) // n_cols # floor division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, width * n_rows // n_cols))
    y_masks, y_tags = y

    # copy of array flattened into one dimension
    axes = axes.flatten() if isinstance(axes, np.ndarray) else (axes,)
    
    # zip - returns zip object which is an iterable of tuples of matched up pairs between original lists
    for xi, yi_mask, yi_tags, ax in zip(x, y_masks, y_tags, axes):
        i = visualizable(xi, yi_mask, thr=thr)
        ax.imshow(i, cmap=plt.cm.gray)
        ax.grid(False)
        if len(yi_tags) == 2:
            title = 'E: {:.1%} - B: {:.1%}'
        elif len(yi_tags) == 4:
            title = 'pE: {:.1%} - pB: {:.1%}\ntE: {:.1%} - tB: {:.1%}'

        ax.text(x=0.5, y=-0.02, s=title.format(*yi_tags), transform=ax.transAxes,
                ha='center', va='top',
                fontsize=width * 4 / 5, fontfamily='monospace')
        ax.set_axis_off()
    
    if out:
        plt.savefig(out, bbox_inches='tight')
        plt.close()

# if __name__ == '__main__':
    # compute_metrics()