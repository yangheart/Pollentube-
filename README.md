
Unveiling Two reaction-diffusion systems with fractional or integral forms through Neural Networks

# Modified PDE-Net Repository

This repository contains the modified PDE-Net.

The fundamental concept of the PDE-Net method, elucidated in \cite{LLD19}, revolves around leveraging a deep convolutional neural network (CNN) to analyze general nonlinear evolution partial differential equations (PDEs) expressed as follows:

$$
\frac{\partial u}{\partial t} = F( z, u, \nabla u, \nabla^2 u,...), \quad z \in \Omega, \, t \in [0, T], \quad \text{(1)}
$$

where $$\(u = u(z, t)\)$$ represents a function (scalar or vector-valued) of the spatial variable $$z$$ and the temporal variable $$t$$. The architecture entails a feed-forward network that combines the forward Euler method in time with the second-order finite difference method in space. This is achieved through the incorporation of specialized filters in the CNN, emulating differential operators. The network is trained to approximate solutions to the aforementioned PDEs, subsequently utilized for making predictions in successive time steps. The findings in [1] affirm the effectiveness of this approach in solving various PDEs, demonstrating commendable accuracy and computational efficiency in comparison to traditional numerical methods.

Within mPDE-Net, we deliberately omit multiplications between derivatives of $$\(u\)$$ and $$\(v\)$$ since such interactions are not commonly observed in biological phenomena. To accommodate interactions in fractional or integral forms, mPDE-Net integrates integral terms and division operations into $$\(SymNet_m^k\)$$, which is used to approximate the multivariate nonlinear response F.
 
We primarily adapted the polypde.py file to align with our equations featuring Neumann boundaries. Additionally, you have the flexibility to customize the library and schemes of $$\(SymNet_m^k\)$$ to better suit your specific equations.
## Usage
| Model 1 | Script | Trained Model |
|----------|----------|----------|
| 2d | 2dsimulation.ipynb | 2d0noise/2d0.05noise|
| 1d | 1dsimulation.ipynb | 1d00noise/1d001noise |

You have two options to obtain the trained model: either retrieve it directly from 2d0noise/2d0.05noise, or 1d00noise/1d001noise, or execute 2dsimulation.ipynb  or 1dsimulation.ipynb  to generate the model. The data for the 2D equation is generated from initcsc2d.py and initb2815.py in the pedtools of aTEAM. As for the 1D equation, the data is sourced from the data file in mfrac-pde-net.

## $L^2$ norm-based term selection criterion and STRidge in \cite{RABK19}
While mPDE-Net demonstrates accurate fitting of data and effective recovery of terms, it occasionally falls short in simplifying the learned PDE, posing challenges in interpretation.
## References

- [LLD19] Long, Z., Lu, Y., & Dong, B. (2019). "PDE-Net 2.0: Learning PDEs from data with a numeric-symbolic hybrid deep network." *Journal of Computational Physics*, 108925.
- [RABK19] S. Rudy, A. Alla, S. L. Brunton, J. N. Kutz, "Data-driven identification of parametric partial differential equations," *SIAM Journal on Applied Dynamical Systems*, vol. 18, no. 2, pp. 643-660, 2019. 
