# SN-NIR
This is the official implementation of **Normal-guided Detail-Preserving Neural Implicit Functions for
High-Fidelity 3D Surface Reconstruction**.

### [Project page](https://sn-nir.github.io/)

<img src="assets/methodology_snnir.jpg">

----------------------------------------
## Installation

## Usage

#### Data Convention

Our data format is inspired from [IDR](https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md) as follows:
```
CASE_NAME
|-- cameras.npz    # camera parameters
|-- normal
    |-- 000.png        # normal map for each view
    |-- 001.png
    ...
|-- albedo
    |-- 000.png        # albedo for each view (optional)
    |-- 001.png
    ...
|-- mask
    |-- 000.png        # mask for each view
    |-- 001.png
    ...
```

One can create folders with different data in it, for instance, a normal folder for each normal estimation method.
The name of the folder must be set in the used `.conf` file.

### Run

**Train**

```shell
python run_experiments.py --mode train --conf ./confs/CONF_NAME.conf --case CASE_NAME
```

**Extract mesh** 

```shell
python run_experiments.py --mode validate_mesh --conf ./confs/CONF_NAME.conf --case CASE_NAME --is_continue
```