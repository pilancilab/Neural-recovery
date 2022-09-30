## Introduction

This repository implements the code for the paper __Overparameterized ReLU Neural Networks Learn

the Simplest Model: Neural Isometry and Phase Transitions__.



Suppose that $\mathbf{X}\in \mathbb{R}^{n\times d}$ and $\mathbf{y}\in \mathbb{R}^d$ are the data matrix and the label vector respectively. In the paper, we focus on the following two-layer neural networks:

- ReLU networks:

  
$$
f^\mathrm{ReLU}(\mathbf{X};\Theta) =(\mathbf{X}\mathbf{W}_1)_+\mathbf{w}_2, \quad \Theta = (\mathbf{W}_1,\mathbf{w}_2),
$$


where $\mathbf{W}_1\in \mathbb{R}^{d\times m}$ and $\mathbf{w}_2\in \mathbb{R}^{m}$.

- ReLU networks with skip connections:

  
$$
f^\mathrm{ReLU}(\mathbf{X};\Theta) =\mathbf{X}\mathbf{w}_{1,1}w_{2,1}+\sum_{i=2}^m (\mathbf{X}\mathbf{w}_{1,i})_+w_{2,i},
$$


where $\Theta = (\mathbf{W}_1,\mathbf{w}_2)$, $\mathbf{W}_1\in \mathbb{R}^{d\times m}$ and $\mathbf{w}_2\in \mathbb{R}^{m}$.

- ReLU networks with normalization layers:


$$
f^\mathrm{ReLU}(\mathbf{X};\Theta) =\sum_{i=1}^m \operatorname{NM}_{\alpha_i}((\mathbf{X}\mathbf{w}_{1,i})_+)w_{2,i},
$$


where $\Theta = (\mathbf{W}_1,\mathbf{w}_2,\mathbf{\alpha})$ and the normalization operation $\operatorname{NM}_\alpha(\mathbf{v})$ is defined by


$$
\operatorname{NM}_{\alpha}(\mathbf{v}) = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}\alpha, \mathbf{v}\in\mathbb{R}^n,\alpha\in \mathbb{R}.
$$



We consider the regularized training problem 


$$
\min_{\Theta} \frac{1}{2}\|f(\mathbf{X};\Theta)-\mathbf{y}\|_2^2+\frac{\beta}{2}R(\Theta).
$$


When $\beta\to 0$, the optimal solution of the above problem solves the following minimal norm problem


$$
    \min_{\Theta} R(\Theta), \text{ s.t. } f(\mathbf{X};\Theta)=\mathbf{y}.
$$



We include code to solve convex optimization formulations of the minimal norm problem and to train nerual networks discussed in the paper, respectively. We also include code to plot the phase transition graphs shown in the paper. 

More details about the numerical experiments can be found in the appendix of the paper.

## Requirements

When solving convex programs, [CVXPY](https://www.cvxpy.org/install/index.html) (version>=1.1.13) is needed. [Mosek](https://www.mosek.com/downloads/) solver is preferred. You can also change the solver according to the documentation of CVXPY.

When training neural networks discussed in the paper, [PyTorch](https://pytorch.org/get-started/locally/) (version>=1.10.0) is needed.

## Usage

Compute the recovery rate of the planted linear neuron by solving the minimal norm problem for ReLU networks with skip connections over 5 independent trials. 
```bash
python rec_rate_skip.py --n 400 --d 100 --sample 5 --sigma 0 --optw 0 --optx 0
```

Compute the absolute distance by solving the convex programs over 5 independent trials. 
```bash
# minimal norm problem
python minnrm_skip.py --n 400 --d 100 --sample 5 --sigma 0 --optw 0 --optx 0

# convex training problem
python cvx_train_skip.py --n 400 --d 100 --sample 5 --sigma 0 --optw 0 --optx 0
```

Compute the test distance by training ReLU networks with skip connections over 10 independent trials. 
```bash
python ncvx_train_skip.py --n 400 --d 100 --sample 10 --sigma 0 --optw 0 --optx 0
```

- You can change `--save_details`, `--save_folder`, `--seed` accordingly. 
- For ReLU networks with normalization layer, you can also set the number of planted neurons by changing `--neu`. Details about the supported types of planted neuron and data matrix can be found in the comments of the code.

## Maintainers
Yixuan Hua (<huayixuan@pku.edu.cn>)

Yifei Wang (wangyf18@stanford.edu)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Citation
If you find this repository helpful, please consider citing:
```

```