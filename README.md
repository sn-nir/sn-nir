# SN-NIR
This is the official implementation of **Normal-guided Detail-Preserving Neural Implicit Function for
High-Fidelity 3D Surface Reconstruction**.

#### [<ins>Aarya Patel</ins>](https://www.linkedin.com/in/aaryapatel007/), [<ins>Hamid Laga</ins>](https://researchportal.murdoch.edu.au/esploro/profile/hamid_laga/overview), and [<ins>Ojaswa Sharma</ins>](https://www.iiitd.ac.in/ojaswa)

### [Project page](https://graphics-research-group.github.io/sn-nir/) | [Paper](https://arxiv.org/abs/2406.04861)

<img src="assets/methodology_snnir.jpg">

----------------------------------------
## Installation

```shell
git clone https://github.com/sn-nir/sn-nir.git
cd sn-nir
pip install -r requirements.txt
```

## Usage

#### Data Convention

Our data format is inspired from [IDR](https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md) as follows:
```
CASE_NAME
|-- cameras.npz    # camera parameters
|-- image
    |-- 000.png        # image for each view
    |-- 001.png
    ...
|-- normal
    |-- 000.png        # normal map for each view
    |-- 001.png
    ...
|-- depth
    |-- 000.png        # depth for each view
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

## Citation
If you find our code useful for your research, please cite as

```
@Article{Patel25,
    title={Normal-guided Detail-Preserving Neural Implicit Function for High-Fidelity 3D Surface Reconstruction},
    author={Patel, Aarya and Laga, Hamid and Sharma, Ojaswa},
    journal = {Proceedings of the ACM on Computer Graphics and Interactive Techniques},
    number = {1},
    volume = {8},
    article = {12},
    month = {May},
    doi = {https://doi.org/10.1145/3728293},
    year={2025}
  }
```


