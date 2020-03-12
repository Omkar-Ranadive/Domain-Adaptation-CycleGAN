import cv2 
import numpy as np 
import os 
from pathlib import Path 
frames_sim = []
frames_real = []
# Change current working directory to Python file's directory 
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
dname = Path(dname)
# Set the directories for simulated and real images 
simulated_dir = dname / 'test' / 'sim'
real_dir = dname  / 'test' / 'real'



# First get the images from the simulator (GTA 5)
for img in sorted(os.listdir(simulated_dir)): 
    frames_sim.append(cv2.imread(str(simulated_dir / img)))

# Then, get the output images (real world generated)
for img in sorted(os.listdir(real_dir)): 
    frames_real.append(cv2.imread(str(real_dir / img)))

# Parameters for video generation 
h, w, layers = frames_sim[0].shape 
size = (w, h) 

# Generate the video for simulated data 
video_sim = cv2.VideoWriter('simulator.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
for img in frames_sim: 
    video_sim.write(img)

video_sim.release()

# Generate the video for real data 
video_real = cv2.VideoWriter('sim2real.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
for img in frames_real: 
    video_real.write(img)

video_real.release()

# Now, do the same for segmented images 
simulated_dir = dname / 'results' / 'sim'
real_dir = dname  / 'results' / 'real'
frames_sim, frames_real = [], []
# First get the images from the simulator (GTA 5)
for img in sorted(os.listdir(simulated_dir)): 
    frames_sim.append(cv2.imread(str(simulated_dir / img)))

# Then, get the output images (real world generated)
for img in sorted(os.listdir(real_dir)): 
    frames_real.append(cv2.imread(str(real_dir / img)))

# Parameters for video generation 
h, w, layers = frames_sim[0].shape 
size = (w, h) 

# Generate the video for simulated data 
video_sim = cv2.VideoWriter('simulator_seg.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
for img in frames_sim: 
    video_sim.write(img)

video_sim.release()

# Generate the video for real data 
video_real = cv2.VideoWriter('sim2real_seg.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
for img in frames_real: 
    video_real.write(img)

video_real.release()
