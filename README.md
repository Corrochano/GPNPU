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

To evaluate performance across different scenarios, matrix multiplication is tested with various matrix sizes such as **100, 1K, 2K, 5K, 7K, 8K, 10K, 15K, or 20K**

### Jacobi
The jacobi method is an iterative algorithm which resolve strictly diagonally dominant differential equation systems. </br>

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

One of the most common uses is resolve the heat equation:
``` math
\frac{\partial u}{\partial t} = \left( \frac{\partial^2 u}{\partial x_1^2} + \frac{\partial^2 u}{\partial x_2^2} + \dots + \frac{\partial^2 u}{\partial x_n^2} \right)
```

I use this approach because... TODO

### Multigrid Jacobi

## Apple

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

