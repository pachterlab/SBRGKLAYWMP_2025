# Code to generate figures for Swarna et al. 2025. 

## Data to generate figures

Data to generate figures can be accessed at the following DOI: 10.5281/zenodo.17654220


## 📘 **Environment Setup & Notebook Usage**

This project uses a reproducible Conda environment defined in `environment.yaml`.
Follow the steps below to install the environment and run the notebooks.

---

## 🔧 1. Install the Conda Environment

Make sure you have **Anaconda** or **Miniconda** installed.

Clone this repository:

```bash
git clone https://github.com/pachterlab/SBRGKLAYWMP_2025.git
cd SBRGKLAYWMP_2025
```

Create the environment from the YAML file:

```bash
conda env create -f environment.yaml
```

---

## ▶️ 2. Activate the Environment

```bash
conda activate swarna2025_env
```

---
## ▶️ 3. Install wompwomp for alluvial plots
```bash
pip install git+https://github.com/pachterlab/wompywompy
```


## 🧠 4. Register the Jupyter Kernel

The `ipykernel` package is already included in the environment.
Register this environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name swarna2025_env --display-name "Python (swarna2025_env)"
```

You will now see a new kernel inside Jupyter Notebook / JupyterLab.

---

## 📓 5. Run the Notebooks

Launch Jupyter:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

**Kernel → Change Kernel → Python (swarna2025_env)**

---

## 🧪 6. Run the Simulation Benchmark Experiments

The simulation benchmark scripts are in `benchmark_code/`. After activating
the environment, run the full five-pattern benchmark with:

```bash
python benchmark_code/run_final_benchmark.py --run-name simulation_run
```

This writes the simulated `.h5ad`, method-specific outputs, combined recovery
CSV, and heatmap into `benchmark_code/results/simulation_run/`.

To run the single-cell versus pseudobulk resolution challenge on the newest
simulation output:

```bash
python benchmark_code/evaluate_singlecell_v_pseudobulk.py
```

Its outputs are saved under the selected run's
`singlecell_vs_pseudobulk/` directory.
