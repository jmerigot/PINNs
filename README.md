# PINNs

## Repository Author: Jules Merigot


*Repository dedicated to the work and research completed during an end-of-study internship at the LOCEAN laboratory of the IPSL-CNRS collaboration located on the Pierre et Marie Curie Campus of Sorbonne University in Paris. The final report was submitted to the MIDO Department of Computer Science of Paris Dauphine University in partial fulfillment of the requirements for the masters degree of Master IASD, Artificial Intelligence and Data Science.*

---

### Abstract

In order to push our research further, we investigated the implementation of a Physics-informed Neural Network (PINN) to improve predictions of physical phenomenons such as SLA and SST. PINNs integrate physical laws, including fluid dynamics and boundary conditions, directly into the learning process. This improves the model's ability to generalize from limited data and ensures that predictions comply with established physical laws and principles. We sought to create a PINN to model a simple, dynamic fluid flow of temperature using physical constraints implemented via a partial differential equation (PDE), with the objective of approaching an improved SLA and SST forecast. By striving to model this PDE along with other notable fluid equations, we furthered our understanding of modeling physical systems. Through this work, we gained significant insight into the inner workings of building and training PINNs for dynamic fluid systems, and set the groundwork for future investigation and testing.

*Keywords*---Physics, Deep Learning, Physics-informed Machine Learning.

---

### Description

The **pinnstorch** file contains the code, tests, and tutorials that were inspected from the PINNs-Torch package created by Bafghi et al. in 2023. The original paper can be found here: https://openreview.net/forum?id=nl1ZzdHpab
The GitHub repository from which this code was pulled can be found here: https://github.com/rezaakb/pinns-torch


