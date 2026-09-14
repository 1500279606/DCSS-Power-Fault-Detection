# DCSS: Dynamic Causal State-Space Learning for Power System Fault Detection


## Overview

This repository provides the official implementation of:

**DCSS: Dynamic Causal State-Space Learning with Reliability-Constrained Test-Time Adaptation for Power System Fault Detection**

Modern power systems exhibit complex operating conditions with changing variable dependencies and distribution shifts. DCSS is a dynamic causal state-space framework that integrates regime-dependent dependency modeling, uncertainty-aware state estimation, innovation-based fault scoring, and reliability-constrained test-time adaptation.

The framework aims to achieve reliable fault detection under diverse operating conditions.


## Framework

DCSS consists of four main components:

- **Dynamic Dependency Modeling**
  
  Learn regime-specific temporal dependency structures to characterize changing system dynamics.


- **Dependency-Guided State-Space Estimation**

  Incorporate dynamic dependency context into latent state evolution with uncertainty-aware estimation.


- **Innovation-Based Fault Scoring**

  Detect faults by measuring deviations between prior and posterior latent states.


- **Reliability-Constrained Test-Time Adaptation**

  Adapt the model during deployment using only reliable normal observations while preventing faulty pattern contamination.



## Datasets

Experiments are conducted on:

- SWaT
- WADI
- PSM
- SMD
- MSL
- PROTECT-90 power system EMT dataset


## Installation

Create the environment:

```bash
conda create -n dcss python=3.10
conda activate dcss
