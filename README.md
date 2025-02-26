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
  * [How to use ANE](#how-to-use-ane)   
  * [Convert to MLCore](#convert-to-mlcore)
  * [Use MLCore](#use-mlcore)  
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

### How to use ANE
In order to use the Apple Neural Engine (ANE) we need to convert our algorithms into a MLCore model, so we need to camouflage our general purpouse algorithms in a machine model archive in order to execute there. </br>

In general, there are thre more problems right there: </br>
  1. What operations works on ANE are not officially public documented, and there are almost not information, but I can [check a little bit](https://github.com/hollance/neural-engine/blob/master/docs/unsupported-layers.md).
  2. We can specify to only run on ANE, so there are things that must be executed on CPU. We need to see what happening with our algorithm because this.
  3. ANE only can run Float16 data.

I needed to test almost all that questions on my just to be sure of the correct running of the different algorithms. </br>

To check if the ANE is on usage, on the first place I used [Asitop](https://github.com/tlkh/asitop) to see the usage on  live time:
![Captura desde 2025-02-25 11-09-41](https://github.com/user-attachments/assets/178ce7c2-162e-4491-9cd0-d646d32ec775)

But to take metrics I used tegrastats with this command:

```shell
sudo powermetrics -i 100 --samplers cpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info --show-process-energy 
```
I used the command saving the output into a file to process it and take only the information that I need. </br>

Finally, to see where is executed each operation, I use XCode just opening the model with it on the file explorer and executing a test.

![image](https://github.com/user-attachments/assets/b69cdfcb-0a05-45dd-91e0-65dc9e1121b7)




### Convert to MLCore
I follow the [Apple documentation](https://apple.github.io/coremltools/docs-guides/source/convert-a-torchvision-model-from-pytorch.html) of how to convert models to MLCore from PyTorch. </br>

As you can see on the link above, there are two forms to convert a pytorch model to MLCore: FromTrace and FromExport. </br>
In my code I used FromTrace method, for example:

```
    traced_model = torch.jit.trace(myModel, (arg1, arg2, arg3))
    jacobi_from_trace = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=arg1.shape, dtype=npfloat), ct.TensorType(shape=arg2.shape, dtype=npfloat), ct.TensorType(shape=arg3.shape, dtype=npfloat)],
        outputs=[ct.TensorType(dtype=npfloat)],
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ctfloat
    )
```
In that code, npfloat is the desire precision (for example, np.float16) and ctfloat is the coreml desire precision (for example, ct.precision.FLOAT16). </br>
Is necessary to specify minimum_deployment_target=ct.target.macOS13 because there are operations that may execute in CPU instead of ANE otherwise.

### Use MLCore
The first step to use MLCore is load the previously saved model

```Python
mlmodel = ct.models.MLModel(f"jacobi{nx}_model_{datatype}_{nt}.mlpackage", compute_units=ct.ComputeUnit.ALL)
```

On that step, we need to specify on what compute units we want to execute the model among this options:
  - ct.ComputeUnit.ALL
  - ct.ComputeUnit.CPU_AND_NE
  - ct.ComputeUnit.CPU_AND_GPU
  - ct.ComputeUnit.CPU_ONLY

The second (and last) step is specify the inputs on a dictionary with the sames names as the model inputs as keys and call the model with it:

```Python
input_dict = {'arg1': arg1, 'arg2': arg2, 'arg3': arg3}
result = mlmodel.predict(input_dict)
```

### Python Dependencies
- torch==2.3.0 
- coremltools==8.0b2 
- torchvision==0.18.0

### M1
All the MLCore models was training on a Mac Mini M1, where all the metrics (time of execution, GFLOPs/s and mean power consumption, that means all the taken mW and calculate the average) was also taken, as they can be on all the devices that I used. </br>
On the next sections, I will show that results for each algorthm.

#### Matrix Multiplication
# Matmul Performance Comparison

|            | |      32x32      |          |                         |
|------------|----------------------------|------------|----------|-------------------------|
| Precision  | Accelerator                | Time of Execution (s) | GFLOPS/s | Mean Total Power (mW) |
| FP32       | ANE                        | <0.0001               | 1.82     | 4991.07                 |
| FP16       | ANE                        | 0.0004                | 0.19     | 4387.39                 |
| FP32       | GPU                        | 0.0004                | 0.17     | 4696.17                 |
| FP16       | GPU                        | 0.0004                | 0.17     | 4670.67                 |
| FP32       | CPU                        | <0.0001               | 1.42     | 4807.27                 |
| FP16       | CPU                        | 0.0001                | 1.11     | 5025.2                  |

|            | |      64x64      |          |                         |
|------------|----------------------------|------------|----------|-------------------------|
| Precision  | Accelerator                | Time of Execution (s) | GFLOPS/s | Mean Total Power (mW) |
| FP32       | ANE                        | <0.0001               | 13.86     | 5188.466667                 |
| FP16       | ANE                        | 0.0004                | 1.43     | 4785.722222                 |
| FP32       | GPU                        | 0.0004                | 1.38     | 4830.888889                 |
| FP16       | GPU                        | 0.0004                | 1.33     | 4799.5                 |
| FP32       | CPU                        | <0.0001               | 10.89     | 5085.6                 |
| FP16       | CPU                        | 0.0001                | 8.02     | 5308.933333                 |


|            | |      128x128      |          |                         |
|------------|----------------------------|------------|----------|-------------------------|
| Precision  | Accelerator                | Time of Execution (s) | GFLOPS/s | Mean Total Power (mW) |
| FP32       | ANE                        | <0.0001               | 88.24    | 5216.866667           |
| FP16       | ANE                        | 0.0004                | 11.27    | 4518.611111           |
| FP32       | GPU                        | 0.0004                | 10.66    | 4728.888889           |
| FP16       | GPU                        | 0.0004                | 10.25    | 4518.333333                 |
| FP32       | CPU                        | 0.0001                | 75.58     | 5063.6666667                 |
| FP16       | CPU                        | 0.0001                | 49.77    | 5115.866667                 |

|            | |      256x256      |          |                         |
|------------|----------------------------|------------|----------|-------------------------|
| Precision  | Accelerator                | Time of Execution (s) | GFLOPS/s | Mean Total Power (mW) |
| FP32       | ANE                        |                |      |                  |
| FP16       | ANE                        |                 |     |                  |
| FP32       | GPU                        |                 |     |                  |
| FP16       | GPU                        |                 |     |                  |
| FP32       | CPU                        |                |      |                  |
| FP16       | CPU                        |                 |     |                  |

|            | |      512x512      |          |                         |
|------------|----------------------------|------------|----------|-------------------------|
| Precision  | Accelerator                | Time of Execution (s) | GFLOPS/s | Mean Total Power (mW) |
| FP32       | ANE                        |                |      |                  |
| FP16       | ANE                        |                 |     |                  |
| FP32       | GPU                        |                 |     |                  |
| FP16       | GPU                        |                 |     |                  |
| FP32       | CPU                        |                |      |                  |
| FP16       | CPU                        |                 |     |                  |

|            | |      1024x1024      |          |                         |
|------------|----------------------------|------------|----------|-------------------------|
| Precision  | Accelerator                | Time of Execution (s) | GFLOPS/s | Mean Total Power (mW) |
| FP32       | ANE                        |                |      |                  |
| FP16       | ANE                        |                 |     |                  |
| FP32       | GPU                        |                 |     |                  |
| FP16       | GPU                        |                 |     |                  |
| FP32       | CPU                        |                |      |                  |
| FP16       | CPU                        |                 |     |                  |

|            | |      2048x2048      |          |                         |
|------------|----------------------------|------------|----------|-------------------------|
| Precision  | Accelerator                | Time of Execution (s) | GFLOPS/s | Mean Total Power (mW) |
| FP32       | ANE                        |                |      |                  |
| FP16       | ANE                        |                 |     |                  |
| FP32       | GPU                        |                 |     |                  |
| FP16       | GPU                        |                 |     |                  |
| FP32       | CPU                        |                |      |                  |
| FP16       | CPU                        |                 |     |                  |

|            | |      4096x4096      |          |                         |
|------------|----------------------------|------------|----------|-------------------------|
| Precision  | Accelerator                | Time of Execution (s) | GFLOPS/s | Mean Total Power (mW) |
| FP32       | ANE                        |                |      |                  |
| FP16       | ANE                        |                 |     |                  |
| FP32       | GPU                        |                 |     |                  |
| FP16       | GPU                        |                 |     |                  |
| FP32       | CPU                        |                |      |                  |
| FP16       | CPU                        |                 |     |                  |

|            | |      4512x4512      |          |                         |
|------------|----------------------------|------------|----------|-------------------------|
| Precision  | Accelerator                | Time of Execution (s) | GFLOPS/s | Mean Total Power (mW) |
| FP32       | ANE                        |                |      |                  |
| FP16       | ANE                        |                 |     |                  |
| FP32       | GPU                        |                 |     |                  |
| FP16       | GPU                        |                 |     |                  |
| FP32       | CPU                        |                |      |                  |
| FP16       | CPU                        |                 |     |                  |

|            | |      4800x4800      |          |                         |
|------------|----------------------------|------------|----------|-------------------------|
| Precision  | Accelerator                | Time of Execution (s) | GFLOPS/s | Mean Total Power (mW) |
| FP32       | ANE                        |                |      |                  |
| FP16       | ANE                        |                 |     |                  |
| FP32       | GPU                        |                 |     |                  |
| FP16       | GPU                        |                 |     |                  |
| FP32       | CPU                        |                |      |                  |
| FP16       | CPU                        |                 |     |                  |

|            | |      5000x5000      |          |                         |
|------------|----------------------------|------------|----------|-------------------------|
| Precision  | Accelerator                | Time of Execution (s) | GFLOPS/s | Mean Total Power (mW) |
| FP32       | ANE                        |                |      |                  |
| FP16       | ANE                        |                 |     |                  |
| FP32       | GPU                        |                 |     |                  |
| FP16       | GPU                        |                 |     |                  |
| FP32       | CPU                        |                |      |                  |
| FP16       | CPU                        |                 |     |                  |

|            | |      8192x8192      |          |                         |
|------------|----------------------------|------------|----------|-------------------------|
| Precision  | Accelerator                | Time of Execution (s) | GFLOPS/s | Mean Total Power (mW) |
| FP32       | ANE                        |                |      |                  |
| FP16       | ANE                        |                 |     |                  |
| FP32       | GPU                        |                 |     |                  |
| FP16       | GPU                        |                 |     |                  |
| FP32       | CPU                        |                |      |                  |
| FP16       | CPU                        |                 |     |                  |

|            | |      16384x16384      |          |                         |
|------------|----------------------------|------------|----------|-------------------------|
| Precision  | Accelerator                | Time of Execution (s) | GFLOPS/s | Mean Total Power (mW) |
| FP32       | ANE                        |                |      |                  |
| FP16       | ANE                        |                 |     |                  |
| FP32       | GPU                        |                 |     |                  |
| FP16       | GPU                        |                 |     |                  |
| FP32       | CPU                        |                |      |                  |
| FP16       | CPU                        |                 |     |                  |

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

