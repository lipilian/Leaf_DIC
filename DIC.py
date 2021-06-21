# %%
import cv2 as cv
import glob
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import trackpy as tp
import pandas as pd
# %%
fileNames = glob.glob('*.mat')
# %%
for fileName in fileNames:
    Name = fileName.split('.mat')
    Name = Name[0]; SaveName = Name + '_rearrange'
    data = io.loadmat(fileName)
    points = [value for key, value in data.items() if 'Centroid' in key ]

    points = points[0]
    print(points.shape)
 
    [m,t,_] = points.shape
    frames = []
    for f in range(t):
        xy_data = points[:,f,:]
        frame_Num = np.ones((m,1)) * f
        frame_Data = np.concatenate((xy_data, frame_Num),axis = 1)
        df = pd.DataFrame(data = frame_Data, columns = ['x', 'y', 'frame'])
        frames.append(df)
    df = pd.concat(frames)


    traj = tp.link(df, 10, memory=1)
    tp.plot_traj(traj)



    firstFrame = traj[traj['frame'] == 0]
    firstFrame = firstFrame.sort_index()
    particleLabel = firstFrame['particle'].tolist()

    for i in range(m):
        label = particleLabel[i]
        particleData = traj[traj['particle'] == label]
        particleData = particleData.sort_values(by = 'frame')
        particleData = particleData.to_numpy()[:,0:2]
        points[i,:,:] = particleData

    data[SaveName] = points
    io.savemat(fileName, data)

# %%
