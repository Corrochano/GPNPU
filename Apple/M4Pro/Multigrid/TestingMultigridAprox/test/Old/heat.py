'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
     
m=2**8-1                           # Number of grid points
h=1./(m+1)                         # Mesh width
x=np.linspace(0,1,m+2); x=x[1:-1]  # grid
alpha=1.; beta=3.                  # boundary values
phi = lambda x: 20.* np.pi * x**3
f = lambda x: -20 + 0.5*120*np.pi*x * np.cos(phi(x)) - 0.5*(60*np.pi*x**2)**2 * np.sin(phi(x)) # RHS

plt.plot(x,f(x)); plt.xlabel('x'); plt.ylabel('f(x)')
plt.savefig("rhs_function_plot.png")  # Save the figure
'''
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation parameters
nx = ny = 1024
dx = dy = 1.0 / (nx - 1)
dt = 0.001
alpha = 0.01
steps = 200

# Grid
x = torch.linspace(0, 1, nx)
y = torch.linspace(0, 1, ny)
X, Y = torch.meshgrid(x, y, indexing='ij')

# Initial condition: Gaussian bump
x0, y0 = 0.5, 0.5
sigma = 0.1
u = torch.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2)).clone()

# For plotting
fig, ax = plt.subplots()
cax = ax.imshow(u.numpy(), cmap='hot', origin='lower', extent=[0, 1, 0, 1])
fig.colorbar(cax)

def update(frame):
    global u
    u_new = u.clone()

    # Finite difference Laplacian (interior only)
    u_new[1:-1, 1:-1] = u[1:-1, 1:-1] + alpha * dt * (
        (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 +
        (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
    )

    u = u_new

    # Update the image
    cax.set_array(u.numpy())
    return [cax]

# Create animation
ani = animation.FuncAnimation(fig, update, frames=steps, interval=100, blit=True)

# Save or display
ani.save("heat_diffusion.gif", writer='pillow')
# plt.show()  # Uncomment if you want to display it interactively

