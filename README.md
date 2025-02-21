# GPNPU  
This repository contains my work with various **neural accelerators** to **evaluate** their performance on **general-purpose algorithms**, such as matrix multiplication and Jacobi methods.  

The project primarily focuses on Apple's ANE and Intel NPUs, utilizing an Apple M1, Apple M4 Pro, and an Intel NUC.  

Additionally, this work serves as my **Master's Final Project** for the **Master’s in Computer Engineering at Complutense University of Madrid**. 

<div style="text-align: center;">
  <img src="https://www.ucm.es/data/cont/docs/3-2016-07-21-EscudoUCMTransparenteBig.png?raw=true" alt="UCM Logo" width="250"/>
</div>

# Author
Álvaro Corrochano López

# Table of Contents
- [Algorithms](#algorithms)  
  * [Matrix Multiplication](#matrix-multiplication)  
  * [Jacobi](#jacobi)  
  * [Multigrid Jacobi](#multigrid-jacobi)  

- [Apple](#apple)  
  * [Python Dependencies](#python-dependencies)  
  * [M1](#m1)  
    + [Matrix Multiplication](#matrix-multiplication-1)  
    + [Jacobi](#jacobi-1)  
    + [Multigrid Jacobi](#multigrid-jacobi-1)
- [Intel NUC](#intel-nuc)  
  * [Matrix Multiplication](#matrix-multiplication-2)  
  * [Jacobi](#jacobi-2)  
  * [Multigrid Jacobi](#multigrid-jacobi-2)  
- [License](#license)
    * [Copyright Notice](#copyright-notice)

## Algorithms
When **GPUs** first entered the market, they **were designed for specific purposes**. However, over time, they have been **increasingly used** in applications like **machine learning**, which they were not originally intended for.  

For this reason, I test various **general-purpose algorithms** to **evaluate** whether **neural accelerators** can achieve **high performance** in fields beyond machine learning.  


### Matrix Multiplication
First, I test matrix multiplication because it is a fundamental operation frequently used in machine learning. As a result, it should be optimized for execution on neural accelerators.  

Matrix multiplication involves multiplying two matrices of compatible sizes. This means the number of columns in the first matrix must be equal to the number of rows in the second matrix. The resulting matrix will have the same number of rows as the first matrix and the same number of columns as the second matrix. 

```math
\text{A} * \text{B} = \text{C}
```

I tested different grid sizes with 100 iterations, including **100×100, 500×500, 1K×1K, 2K×2K, 3K×3K, 5K×5K, 7K×7K, 8K×8K, 10K×10K, 12K×12K, 15K×15K, and 20K×20K**.  

### Jacobi
The Jacobi method is an **iterative algorithm** used to solve strictly diagonally dominant differential equation systems. </br> 

The pseudocode can be described as follows:  

The pseudocode must be describe as:

``` Pseudocode
k = 0
while convergence not reached do
    for i := 1 step until n do
        σ = 0
        for j := 1 step until n do
            if j ≠ i then
                σ = σ + aij xj(k)
            end
        end
        xi(k+1) = (bi − σ) / aii
    end
    increment k
end
```

One of the most common applications of the Jacobi method is solving the **heat equation**:
``` math
\frac{\partial u}{\partial t} = \left( \frac{\partial^2 u}{\partial x_1^2} + \frac{\partial^2 u}{\partial x_2^2} + \dots + \frac{\partial^2 u}{\partial x_n^2} \right)
```

I chose this equation because of its relevance and to take the challenge to the next level by working with something more complex than before. </br>

I tested different grid sizes with 100 iterations, including **100×100, 500×500, 1K×1K, 2K×2K, 3K×3K, 5K×5K, 7K×7K, 8K×8K, 10K×10K, 12K×12K, 15K×15K, and 20K×20K**.  

### Multigrid Jacobi
The Multigrid Method is **another algorithm** designed to solve differential equations using a hierarchy of discretizations. It is particularly useful for problems that exhibit multiple scales of behavior. </br>  

There are different approaches to the Multigrid Method, but I use the most common one, known as the **V-Cycle Multigrid**. </br>

Here is the pseudocode for this algorithm: 

``` Pseudocode
function phi = V_Cycle(phi,f,h)
    % Recursive V-Cycle Multigrid for solving the Poisson equation (\nabla^2 phi = f) on a uniform grid of spacing h

    % Pre-Smoothing
    phi = smoothing(phi,f,h);

    % Compute Residual Errors
    r = residual(phi,f,h);

    % Restriction
    rhs = restriction(r);

    eps = zeros(size(rhs));

    % stop recursion at smallest grid size, otherwise continue recursion
    if smallest_grid_size_is_achieved
        eps = coarse_level_solve(eps,rhs,2*h);
    else
        eps = V_Cycle(eps,rhs,2*h);
    end

    % Prolongation and Correction
    phi = phi + prolongation(eps);

    % Post-Smoothing
    phi = smoothing(phi,f,h);
end
```

to also solve the heat equation, that is described on the [jacobi section](#jacobi).

## Apple

One of the more innovated parts of this project is use the Apple Neuronal Accelerator (ANE), that are barely documented. </br>

To use the ANE, we need to use the general purpouse algoritmhs into Neuronal Networks, so I use PyTorch to create a "model" that inside only performs the algorithm and then I convert it to the Apple format that is called mlpackage. </br>

We can do that with the Python library coremltools. I explain it with more detail in his own section.

### Python Dependencies
- torch==2.3.0 
- coremltools==8.0b2 
- torchvision==0.18.0

### M1

#### Matrix Multiplication

#### Jacobi

#### Multigrid Jacobi

## Intel NUC
TODO

### Matrix Multiplication
TODO

### Jacobi
TODO

### Multigrid Jacobi
TODO

## License
This project is licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

### Copyright Notice
Copyright 2025 Álvaro Corrochano López

