import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

img = plt.imread('./results/original_frames/00000.jpg')
fig, ax = plt.subplots(1)
ax.imshow(img)
x1, y1, x2, y2 = 230, 283, 352, 407
rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.savefig('bbox.png')
