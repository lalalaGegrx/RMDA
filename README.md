# Low-Dimensional Matrix Discriminant Analysis of EEG Network Connectivity on Riemannian Manifolds for Emotion Recognition

## Introduction



### Requirements

- MATLAB R2019a (or other versions)

### How to run

1. Create the `.\Dataset` folder in the root directory and a Dataset folder in `.\Dataset`, such as `.\Dataset\DEAP`.
2. Run `.\Dataset\Generate_*.m` to create either covariance, coherence, or PSD matrices.
3. Run `.\main_LeaveOne*Out_*.m` to get the results of leave-one-trial-out and leave-one-subject-out emotion recognition.
