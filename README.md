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
Firstly, I test matrix multiplication because is a basic operator that is used frequently on machine learning, so it must be quick to perform on this neuronal accelerators. </br>

Matrix multiplication consists on TODO

### Jacobi

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

