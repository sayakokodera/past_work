# taken from https://stackoverflow.com/questions/50505550/plot-mouse-movement-python

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# load reference reco
#trajectory_img = Image.open('D:/Python/PycharmTFSTest/UltrasoundRekoBibPython/Codebase/2018_10_04_SHMNDT_Contribution_Assistix_SAFT_Krieg/Code/trajectory_3_800.png')
#trajectory_img = Image.open('/Users/sako5821/Desktop/git/2018_Sayako_Kodera_BA/Code/scan_path_img/manual_scan_1.png')
#trajectory_arr = np.array(trajectory_img)
trajectory_arr = np.random.randn(100,100)


fig, ax = plt.subplots()

#ax.set_xlim(0, 150-1)
#ax.set_ylim(0, 55-1)
ax.imshow(trajectory_arr)


x,y = [0], [0]
# create empty plot
points, = ax.plot([], [], 'o')

# cache the background
background = fig.canvas.copy_from_bbox(ax.bbox)

def on_move(event):
    # append event's data to lists
    x.append(event.xdata)
    y.append(event.ydata)
    # update plot's data
    points.set_data(x,y)
    # restore background
    fig.canvas.restore_region(background)
    # redraw just the points
    ax.draw_artist(points)
    # fill in the axes rectangle
    fig.canvas.blit(ax.bbox)


fig.canvas.mpl_connect("motion_notify_event", on_move)
plt.show()

# as soon as the plot window is closed store points as list to disk
point_list = np.array([x,y]).T
# remove invalid entries
valid_indices = np.squeeze(np.array(np.where(point_list[:,0] != None)))
point_list = point_list[valid_indices,:]
# remove duplicate entries?

# write to disk
np.save('trajectory',point_list)