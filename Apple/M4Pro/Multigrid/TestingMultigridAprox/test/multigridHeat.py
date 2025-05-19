#Copyright 2024 Álvaro Corrochano López
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import coremltools as ct
import torch.nn.functional as F
from matplotlib import cm
import imageio

# Multigrid
class JacobiMachine(nn.Module):
    def __init__(self, nt=1000, num_levels=9, datatype=torch.float32):
        super(JacobiMachine, self).__init__()
        self.datatype=datatype
        self.num_levels = num_levels
        self.nt = torch.tensor(nt, dtype=self.datatype)
        self.snaps = []

    def restriction(self, residual): # Applied when we want to convert from finegrid to coarsergrid
        return nn.AvgPool2d(kernel_size=2)(residual)
    
    def interpolate(self, f, target_size): # Applied when we want to convert from coarsergrid to finegrid
        return F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)

    def jacobi(self, Z, mask): # Jacobi method   
      x = Z
      x_prev = x.clone()
              
      # Define the 3x3 kernel
      kernel = torch.tensor([[0.0, 0.25, 0.0],
                            [0.25, 0.0, 0.25],
                            [0.0, 0.25, 0.0]], dtype=self.datatype).view(1, 1, 3, 3)
                            
      i = torch.tensor(0, dtype=self.datatype)
      
      print(mask.shape  )
      
      while torch.ne(i, self.nt):
          x_prev = x.clone()
          x_next = F.conv2d(x_prev, kernel, padding=1)            
          x = x_next * mask + x_prev * (1 - mask)
          
          i = torch.add(i, 1)
      
      return x

    def forward(self, X, Mask1, Mask2, Mask3, Mask4, Mask5, Mask6, Mask7, Mask8, Mask9, Mask10):
        snapshots = []
        Masks = [Mask1,Mask2,Mask3,Mask4,Mask5,Mask6,Mask7,Mask8,Mask9,Mask10]
        x = X
        
        # Define the 3x3 kernel
        kernel = torch.tensor([[0.0, 0.25, 0.0],
                               [0.25, 0.0, 0.25],
                               [0.0, 0.25, 0.0]], dtype=self.datatype).view(1, 1, 3, 3)
        
        # Define num_levels
        num_levels = self.num_levels
        
        grids = [x]
        
        current_mask = Masks[0]
        
        # Downward phase
        for level in range(1, num_levels):
            masked_input = torch.mul(grids[level - 1] , (Masks[level - 1] > 0))
            unmasked_input = torch.mul(grids[level - 1] , (1 - Masks[level - 1]))

            masked_output = F.conv2d(masked_input, kernel, padding=1)
            unmasked_output = F.conv2d(unmasked_input, kernel, padding=1)

            residual = masked_output + unmasked_output
            
            coarse_residual = self.restriction(residual)  # Restrict to coarser grid
        
            grids.append(coarse_residual.clone())  # Store
            
            # Update graph
            snapshots.append(coarse_residual.clone())
        
        # Solve on the coarsest grid
        coarse_solution = grids[-1]
        coarse_solution = self.jacobi(coarse_solution, Mask9)  # Solve the classic jacobi there
        grids[-1] = coarse_solution  # Update store solution
        
        # Update graph
        snapshots.append(coarse_solution.clone())
               
        # Upward phase
        for level in range(num_levels - 2, -1, -1):

            target_size = grids[level].shape[-2:]
          
            fine_solution = self.interpolate(grids[level + 1], target_size=target_size)  # Interpolate to finer gridclear
                        
            fine_solution = torch.add( fine_solution, grids[level] ) # += grids[level]  # Add correction to finer grid    
            
            masked_input = torch.mul(fine_solution , (Masks[level] > 0))
            unmasked_input = torch.mul(fine_solution , (1 - Masks[level]))

            masked_output = F.conv2d(masked_input, kernel, padding=1)
            unmasked_output = F.conv2d(unmasked_input, kernel, padding=1)

            fine_solution = masked_output + unmasked_output
            
            # Update graph
            snapshots.append(fine_solution.clone())
            
        # Final refinement on the finest grid        
        while len(grids) > 1:
            grids.pop()
        
        final_solution = grids.pop()
        final_solution = self.jacobi(final_solution, Mask1)  # Final Jacobi iterations
        
        # Update graph
        snapshots.append(final_solution.clone())
        
        self.snaps = snapshots      
    
        return final_solution

# Defining our problem
datatype = "fp16"
ctfloat = ct.precision.FLOAT16
npfloat = np.float16
torchfloat = torch.float16
nodes = 1024

# Initialization 
jacobiModel = JacobiMachine(datatype=torchfloat)
jacobiModel.eval()
jacobiModel.float()

dx = nodes
dy = nodes

u = np.zeros((nodes, nodes)) + 20 # Plate is initially as 20 degres C

# Boundary Conditions 

u[0, :] = np.linspace(0, 100, nodes)
u[-1, :] = np.linspace(0, 100, nodes)

u[:, 0] = np.linspace(0, 100, nodes)
u[:, -1] = np.linspace(0, 100, nodes)

# Create masks
u = torch.tensor(u, dtype=torchfloat).unsqueeze(0).unsqueeze(0)

mask = torch.ones_like(u, dtype=torchfloat)
mask[:, :, 0, :] = 0
mask[:, :, -1, :] = 0
mask[:, :, :, 0] = 0
mask[:, :, :, -1] = 0    

# Define num_levels
num_levels = 9

masks = [mask]

for _ in range(num_levels):# precalculate masks
    masks.append(nn.AvgPool2d(kernel_size=2)(masks[-1]).to(torchfloat))    

# Visualizing with a plot
#fig, axis = plt.subplots()

#pcm = axis.pcolormesh(u.squeeze().detach().cpu().numpy(), cmap=plt.cm.jet, vmin=0, vmax=100)
#plt.colorbar(pcm, ax=axis)

# Testing
output = jacobiModel(u, masks[0], masks[1], masks[2], masks[3], masks[4], masks[5], masks[6], masks[7], masks[8], masks[9])

'''
# Visualize
for snap in jacobiModel.snaps:
    plt.imshow(snap.squeeze().cpu().numpy(), cmap='jet', vmin=0, vmax=100)
    plt.colorbar()
    plt.pause(0.1)
    plt.clf()
'''
    
# Convertir los snapshots a imágenes del mismo tamaño
target_shape = jacobiModel.snaps[0].shape[-2:]  # (H, W)
colormapped_frames = []

for snap in jacobiModel.snaps:
    # Asegurarse de que tiene 4 dimensiones (N, C, H, W)
    if snap.dim() == 2:
        snap = snap.unsqueeze(0).unsqueeze(0)
    elif snap.dim() == 3:
        snap = snap.unsqueeze(0)
    
    # Resize al tamaño objetivo
    snap_resized = F.interpolate(snap, size=target_shape, mode='bilinear', align_corners=False)

    # Convertir a NumPy
    snap_np = snap_resized.squeeze().detach().cpu().numpy()
    
    # Normalizar y aplicar colormap
    norm_frame = snap_np / 100.0
    color_frame = cm.jet(norm_frame)[:, :, :3]  # RGBA → RGB
    color_frame = (color_frame * 255).astype(np.uint8)
    
    colormapped_frames.append(color_frame)

# Guardar como GIF
imageio.mimsave("multigrid_evolution.gif", colormapped_frames, fps=5)
print("GIF guardado como multigrid_evolution.gif")

first_img = jacobiModel.snaps[0].squeeze().detach().cpu().numpy()
last_img = jacobiModel.snaps[-1].squeeze().detach().cpu().numpy()

# Guardar como PNG con colormap
plt.imsave('multigrid_first.png', first_img, cmap='jet', vmin=0, vmax=100)
plt.imsave('multigrid_last.png', last_img, cmap='jet', vmin=0, vmax=100)
    
