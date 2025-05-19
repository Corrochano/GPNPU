# https://github.com/Younes-Toumi/Youtube-Channel/blob/main/Simulation%20with%20Python/Heat%20Equation/heatEquation2D.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio

# Defining our problem

a = 110
length = 50 #mm
nodes = 1024

# Initialization 

dx = nodes
dy = nodes

dt = min(   dx**2 / (4 * a),     dy**2 / (4 * a))

u = np.zeros((nodes, nodes)) + 20 # Plate is initially as 20 degres C

# Boundary Conditions 

u[0, :] = np.linspace(0, 100, nodes)
u[-1, :] = np.linspace(0, 100, nodes)

u[:, 0] = np.linspace(0, 100, nodes)
u[:, -1] = np.linspace(0, 100, nodes)

# Visualizing with a plot

fig, axis = plt.subplots()

pcm = axis.pcolormesh(u, cmap=plt.cm.jet, vmin=0, vmax=100)
plt.colorbar(pcm, ax=axis)

# Simulating

counter = 0

snapshots = []

while counter < 1000 :

    w = u.copy()

    for i in range(1, nodes - 1):
        for j in range(1, nodes - 1):

            dd_ux = (w[i-1, j] - 2*w[i, j] + w[i+1, j])/dx**2
            dd_uy = (w[i, j-1] - 2*w[i, j] + w[i, j+1])/dy**2

            u[i, j] = dt * a * (dd_ux + dd_uy) + w[i, j]

    counter += 1
    snapshots.append(u.copy())

    #print("t: {:.3f} [iters], Average temperature: {:.2f} Celcius".format(counter, np.average(u)))

colormapped_frames = []
for snap in snapshots:
    # Normalizar y aplicar colormap
    norm_frame = snap / 100.0
    color_frame = cm.jet(norm_frame)[:, :, :3]  # RGBA → RGB
    color_frame = (color_frame * 255).astype(np.uint8)
    
    colormapped_frames.append(color_frame)

# Guardar como GIF
imageio.mimsave("normal_evolution.gif", colormapped_frames, fps=5)
print("GIF guardado como normal_evolution.gif")

# Guardar como PNG con colormap
plt.imsave('yt_first.png', colormapped_frames[0], cmap='jet', vmin=0, vmax=100)
plt.imsave('yt_last.png', colormapped_frames[-1], cmap='jet', vmin=0, vmax=100)


