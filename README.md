# Low-Dimensional Matrix Discriminant Analysis of EEG Network Connectivity on Riemannian Manifolds for Emotion Recognition

## Introduction

![Illustration of the proposed Riemannian Matrix Discriminant Analysis (RMDA) method. (A) The construction of EEG network connectivity from multi-channel EEG using covariance, coherence, and CPSD matrices. (B) The scheme of the RMDA method makes the two classes more easily separable in the lower-dimensional Riemannian manifold. (C) Flowchart of the complete RMDA method.](methods.jpg)

## How to run

### Requirements

- MATLAB R2019a (or other versions)

### Setup

1. Create the `.\Dataset` folder in the root directory and a Dataset folder in `.\Dataset`, such as `.\Dataset\DEAP`. The DEAP datastet is available at http://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html.
2. Run `.\Dataset\Generate_*.m` to create either covariance, coherence, or CPSD matrices.
3. Run `.\main_LeaveOne*Out_*.m` to get the results of leave-one-trial-out and leave-one-subject-out emotion recognition.

## Citation
```
@article{fang2024emotion,
  title={Emotion recognition from eeg network connectivity using low-dimensional discriminant analysis on Riemannian manifolds},
  author={Fang, Hao and Wang, Mingyu and Wang, Yueming and Yang, Yuxiao},
  journal={Authorea Preprints},
  year={2024},
  publisher={Authorea}
}
