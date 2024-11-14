# Quantum Lattice-Boltzmann Method
Supplementary code for the publication: "A Quantum Algorithm for the Advection-Diffusion Equation in the Lattice-Boltzmann Method". https://www.sciencedirect.com/science/article/pii/S0010465524002960

## Abstract
We present a versatile and efficient quantum algorithm based on the Lattice Boltzmann method (LBM) approximate solution of the linear advection-diffusion equation (ADE). We emphasize that the LBM approximation modifies the diffusion term of the underlying exact ADE and leads to a modified equation (mADE). Due to its versatility in terms of operator splitting, the proposed quantum LBM algorithm for the mADE provides a building block for future quantum algorithms to solve the linearized Navier-Stokes equation on quantum computers. We split the algorithm into four operations: initialization, collision, streaming, and calculation of the macroscopic quantities. We propose general quantum building blocks for each operator, which adapt intrinsically from the general three-dimensional case to smaller dimensions and apply to arbitrary lattice-velocity sets. Based on (sub-linear) amplitude data encoding, we propose improved initialization and collision operations with reduced complexity and efficient sampling-based simulation. Quantum streaming algorithms are based on previous developments. The proposed quantum algorithm allows for the computation of successive time steps, requiring full state measurement and reinitialization after every time step. It is validated by comparison with a digital implementation and based on analytical solutions in one and two dimensions. Furthermore, we demonstrate the versatility of the quantum algorithm for two cases with non-uniform advection velocities in two and three dimensions. Various velocity sets are considered to further highlight the flexibility of the algorithm. We benchmark our optimized quantum algorithm against previous methods employed in sampling-based quantum simulators. We demonstrate sampling efficiency, with sampling accelerated convergence requiring fewer shots.

## Corresponding Author
David Wawrzyniak

Correspondence via 
[mail](mailto:david.wawrzyniak@tum.de).

## Citation
```
@article{WAWRZYNIAK2025109373,
title = {A quantum algorithm for the lattice-Boltzmann method advection-diffusion equation},
journal = {Computer Physics Communications},
volume = {306},
pages = {109373},
year = {2025},
issn = {0010-4655},
doi = {https://doi.org/10.1016/j.cpc.2024.109373},
url = {https://www.sciencedirect.com/science/article/pii/S0010465524002960},
author = {David Wawrzyniak and Josef Winter and Steffen Schmidt and Thomas Indinger and Christian F. Janßen and Uwe Schramm and Nikolaus A. Adams},
}
```

## License
