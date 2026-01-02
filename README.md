# Estimation of time series by Maximum Mean Discrepancy

Matlab implementation of time series estimation by Maximum Mean Discrepancy based on the paper:

*Estimation of time series by Maximum Mean Discrepancy* by Pierre Alquier, Jean-David Fermanian and Benjamin Poignard.

Link: 

# Overview

The code in this replication includes:

- The different models considered in the paper and their estimation by MMD: the file *simulations.m* provides an overview of ISMMD, PSMMD, and alternative estimation methods, for all the models (SV, GARCH, ARMA, non-linear MA and Ricker); the replicator should execute program *simulations.m* for one batch.
- The full experiments (100 batches) can be found in *simulation_SV.m* for SV model, *simulation_GARCH.m* for GARCH model, *simulation_arma.m* for ARMA model, *simulation_NLts.m* for non-linear MA model, *simulation_ricker.m* for Ricker model.
- The code for the selection of an optimal lag p can be found in *validation.m*
- The code comparing the performances of the stochastic gradient descent and the gradient descent based on simulated innovations drawn once only can be found in *simulation_arma.m* and *simulation_NLts.m* for the ARMA model and the non-linear model, respectively.

# Software requirements

The Matlab code was run on a Windows-Intel(R) Xeon(R) Gold 6242R CPU @ 3.10GHz (3.09 GHz) and 128 GB Memory. The version of the Matlab software on which the code was run is a follows: 23.2.0.2859533 (R2023b) Update 10.

The following toolboxes should be installed:

- Global Optimization Toolbox, Version 23.2
- Optimization Toolbox, Version 23.2
- Parallel Computing Toolbox, Version 23.2

The Parallel Computing Toolbox is recommended to run the code in the case of 100 batches-based experiments.

# Description of the code

The main functions to conduct the estimation of ISMMD and PSMMD estimator $\tilde{\theta}^{(k)}_{N,T}$ (ISMMD for $k=1$; PSMMD for $k=2$) for GARCH model are *gradient_garch_ismmd.m* and *gradient_garch_psmmd.m*, respectively.
All the other models (SV, ARMA, non-linear MA and Ricker) are built the same way.

The code for rbf_dot.m, mmd.m and median_heuristic.m were downloaded from the website: https://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm
"A Kernel Two-Sample Test" by Arthur Gretton, Karsten Borgwardt, Malte Rasch, Bernhard Schoelkopf, Alex Smola

# To come

A package that will allow the user to specify the function to be optimized and the constraints to be satisfied will soon be provided. 
