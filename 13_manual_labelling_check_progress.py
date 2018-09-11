import os
import re

folder = 'images1'

files = os.listdir(folder)
images = 0
bboxes = 0
coords = 0
for f in files:
    if f.endswith('.jpg'):
        images += 1
    if f.endswith('bboxes.tsv'):
        bboxes += 1
    if f.endswith('bboxes.labels.tsv'):
        coords += 1

print('Total number of images: %i' %images)
print('Progress on bboxes: {} ({:.1f} %)'.format(bboxes, bboxes/images*100))
print('Progress on coords: {} ({:.1f} %)'.format(coords, coords/images*100))
