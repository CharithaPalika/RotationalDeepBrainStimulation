# Rotational DBS and Decision-Making

This repository explores a simple but interesting idea:

> What if deep brain stimulation did not stimulate the same STN region all the time, but moved across sectors in a rotating pattern?

The project compares **standard DBS** with **sector/rotational DBS** in a computational basal ganglia model of Parkinsonian decision-making.

## DBS Protocols

**Standard DBS (sDBS)** applies stimulation from a fixed electrode-centered field. In the model, the same Gaussian stimulation spread is delivered to the STN throughout the simulation.

**Sector DBS / Rotational DBS (rDBS)** uses the same stimulation pulse and spatial spread, but activates only one STN sector at a time. The active sector changes sequentially, so stimulation rotates across four regions of the STN grid.

In code:

```text
standard DBS        = DBS: True, sector_wise: False
sector/rotational DBS = DBS: True, sector_wise: True
```

## Project Aim

Parkinson's disease is modeled here as two interacting problems:

1. **Value-learning impairment** from reduced dopamine-dependent temporal-difference learning.
2. **Exploration impairment** from pathological synchrony in the STN-GPe network.

The aim is to test whether changing the spatial pattern of DBS can better disrupt pathological STN-GPe synchrony, restore exploratory behavior, and improve reward-guided decision-making.

The model compares four conditions:

```text
Normal -> Parkinsonian -> Standard DBS -> Sector/Rotational DBS
```

The key question is not only whether DBS changes neural activity, but whether that change helps the model make better choices.

## What Is Inside

```text
stn_gpe/        STN-GPe spiking network, DBS waveforms, connectivity, and analysis
basal_ganglia/  Basal ganglia action-selection and reinforcement-learning model
envs/           Decision-making environments
params/         YAML parameter files for network and task conditions
simulations/    Jupyter notebooks for running experiments
results/        Generated data, plots, and final figures
temp/           Temporary YAML files written during simulations
```

## Core Model Pieces

- **STN-GPe network:** a 16 x 16 STN grid and a 16 x 16 GPe grid using Izhikevich spiking neurons.
- **DBS stimulation:** biphasic stimulation pulses with Gaussian spatial spread over the STN.
- **Standard DBS:** a fixed stimulation field centered on the STN grid.
- **Rotational DBS:** a sector-wise stimulation field that rotates across four STN regions.
- **Basal ganglia decision model:** direct and indirect pathways feed a GPi race model for action selection.
- **Learning:** chosen actions are updated using a temporal-difference error signal.

## Decision-Making Tasks

The simulations evaluate behavior in:

- **Iowa Gambling Task:** ambiguous long-term reward and punishment tradeoffs.
- **Non-stationary bandits:** reward contingencies change once during the task.
- **Restless bandits:** reward values drift continuously over time.

These tasks are used to test whether restoring STN-GPe variability and reducing pathological synchrony can improve exploration and reward-guided choices.

## Main Results

In the simulations, Parkinsonian STN-GPe dynamics show strong synchrony and reduced variability. Standard DBS reduces some pathological activity, but the network can still retain residual synchronized structure.

Sector/rotational DBS restores the STN-GPe network closer to the normal-like state:

- lower pathological synchrony
- higher entropy
- higher cluster variability
- more exploratory choice behavior
- better task performance than standard DBS in the tested decision-making tasks

The behavioral benefit is strongest when value learning is not too severely impaired. In other words, restoring exploration helps most when the model can still learn useful action values.

## Important Parameter Files

```text
params/stn_gpe_params/params_Normal.yaml       Normal-like STN-GPe state
params/stn_gpe_params/params_PD.yaml           Parkinsonian STN-GPe state
params/stn_gpe_params/params_std_DBS.yaml      Standard DBS condition
params/stn_gpe_params/params_sector_DBS.yaml   Rotational/sector-wise DBS condition
```

Task-specific parameters live in:

```text
params/decision_making_task_params/
```

## Running the Code

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate light_house_dbs
```

If any notebook reports missing packages, install the analysis dependencies used by the code:

```bash
pip install scipy pyyaml seaborn tqdm
```

Most experiments are organized as notebooks:

```text
simulations/stn_gpe_system/
simulations/decision_making_tasks/
results/decision_making_tasks/
```

A typical workflow is:

1. Run or inspect the STN-GPe condition notebooks.
2. Generate STN activity for normal, PD, standard DBS, or sector-wise DBS.
3. Feed STN activity into the basal ganglia decision model.
4. Compare task performance, choice variability, Q-value error, synchrony, entropy, and cluster variability.

## Naming Note

This repository uses both names:

```text
Rotational DBS = the scientific idea
Sector-wise DBS = the implementation detail
```

They refer to the same stimulation strategy in this project.

## Current Status

This repository includes model components, simulation notebooks, parameter files, intermediate outputs, and final plots. It is intended for understanding, reproducing, and extending the computational experiments.