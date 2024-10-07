# PINNs

## Repository Author: Jules Merigot


*Repository dedicated to the research and work completed during an end-of-study internship at the LOCEAN laboratory of the IPSL-CNRS collaboration located on the Pierre et Marie Curie Campus of Sorbonne University in Paris. The final report was submitted to the MIDO Department of Computer Science of Paris Dauphine University in partial fulfillment of the requirements for the masters degree of Master IASD, Artificial Intelligence and Data Science.*

---

### Abstract

In order to push our research further, we investigated the implementation of a Physics-informed Neural Network (PINN) to improve predictions of physical phenomenons such as SLA and SST. PINNs integrate physical laws, including fluid dynamics and boundary conditions, directly into the learning process. This improves the model's ability to generalize from limited data and ensures that predictions comply with established physical laws and principles. We sought to create a PINN to model a simple, dynamic fluid flow of temperature using physical constraints implemented via a partial differential equation (PDE), with the objective of approaching an improved SLA and SST forecast. By striving to model this PDE along with other notable fluid equations, we furthered our understanding of modeling physical systems. Through this work, we gained significant insight into the inner workings of building and training PINNs for dynamic fluid systems, and set the groundwork for future investigation and testing.

*Keywords*---Physics, Deep Learning, Physics-informed Machine Learning.

---

### Description

The **pinnstorch** folder contains the code, tests, and tutorials that were copied from the PINNs-Torch package created by Bafghi et al. for their NeurIPS paper in 2023, and from which the results were ***correctly*** reproduced for research purposes.
- The original paper can be found here: https://openreview.net/forum?id=nl1ZzdHpab
- The GitHub repository from which this code was pulled can be found here: https://github.com/rezaakb/pinns-torch


The **NavierStokes** folder contains notebooks and files copied from the original PINNs paper by Raissi et al. from 2019, and from which the results were attempted to be reproduced for research purposes. While promising, the results were not well reproduced.
- The original paper can be found here: https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125
- The GitHub repository from which this code was pulled can be found here: https://github.com/maziarraissi/PINNs


The **Burgers** folder contains notebooks and files inspired from the original PINNs paper by Raissi et al. from 2019. This code was inspired by the *NavierStokes* code but was self-adapted for Burger's Equation instead for research purposes. The data is therefore generated within the code based on the PDE and the boundary conditions. The results were promising but not very conclusive.
- The original paper and can can be found at the same location as for the **NavierStokes** folder.


The **TempAdvection_PDE_PINN** file is a notebook containing the initial test code for a simple 2D temperature advection PDE that governs the dynamic fluid flow of a temperature field through a canal. A breakdown of this problem setup can be found in the associated Physics-informed Neural Networks chapter of the Internship Report in this repository.
- Includes a finite-difference method as a numerical solver to establish a semblance of a ground truth and visualize the expected results.
- Then introduces the PINN model with a simple MLP made to solve the PDE. This is done by generating both PDE data and boundary condition data for the model to learn by computing a physics-informed loss and a data-informed loss, respectively. A more in-depth description of this can be once again found in the final report.
