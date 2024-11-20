# NetFormer

## Set up environment

```bash
conda env create --name NetFormer --file environment.yml
```

Install NetFormer package in root directory

```bash
conda activate NetFormer
pip install -e .
```

## Datasets

* Simulation Data
  * Nonlinear and Nonstationary System Simulation
  * Spike-Timing-Dependent Plasticity (STDP) Simulation
  * Connectivity-Constrained Simulation
* Neural Data
  * [Bugeon et al. 2022, Nature](https://www.nature.com/articles/s41586-022-04915-7) (A transcriptomic axis predicts state modulation of cortical interneurons), download dataset from this [link](https://figshare.com/articles/dataset/A_transcriptomic_axis_predicts_state_modulation_of_cortical_interneurons/19448531).
 
## Connectivity-Constrained Simulation

```bash
cd scripts
bash train_NetFormer_sim.sh
```
## Neural Data

```bash
cd scripts
bash train_NetFormer_mouse.sh
```

## Citations
