# Learning Based Dynamic Cluster Reconfiguration for UAV Mobility Management with 3D Beamforming

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Irshadmeer/ICMLCN-2024-Dynamic-Clustering)
![GitHub](https://img.shields.io/github/license/Irshadmeer/ICMLCN-2024-Dynamic-Clustering)
[![DOI](https://img.shields.io/badge/doi-10.1109/ICMLCN59089.2024.10625071-informational)](https://doi.org/10.1109/ICMLCN59089.2024.10625071)
[![arXiv](https://img.shields.io/badge/arXiv-2402.00224-informational)](https://arxiv.org/abs/2402.00224)


This repository is accompanying the paper "Learning Based Dynamic Cluster
Reconfiguration for UAV Mobility Management with 3D Beamforming" (I. Meer,
K.-L. Besser, M. Ozger, D. Schupke, V. Poor, C. Cavdar. In Proceedings of the
2024 IEEE International Conference on Machine Learning for Communication and
Networking (ICMLCN), pp. 486-491, May 2024),
[doi:10.1109/ICMLCN59089.2024.10625071](https://doi.org/10.1109/ICMLCN59089.2024.10625071),
[arXiv:2402.00224](https://arxiv.org/abs/2402.00224).

The idea is to give an interactive version of the calculations and presented
concepts to the reader. One can also change different parameters and explore
different behaviors on their own.


## File List
The following files are provided in this repository:

- `baseline.py`: Python module that implements the baseline algorithms for
  comparison
- `beamforming.py`: Python module that implements the 3D beamforming
- `data_logger.py`: Python module that contains the callback to store data
  during training
- `environment.py`: Python module that contains the gym environment of the
  considered scenario
- `main_training.py`: Main Python script that starts the training and saves the
  model
- `movement.py`: Python module that implements the UAV movement based on SDEs
- `plot_test.py`: Python script that plots the test results
- `reliability.py`: Python module that contains the calculations of the outage
  probability
- `test.py`: Python script that runs the test phase
- `util.py`: Python module that contains utility functions.
- `requirements.txt`: File listing all the required libraries
## Usage
### Running it online
You can use services like [CodeOcean](https://codeocean.com) or
[Binder](https://mybinder.org/v2/gh/Irshadmeer/ICMLCN-2024-Dynamic-Clustering/HEAD)
to run the scripts online.

### Local Installation
If you want to run it locally on your machine, Python3 and Jupyter are needed.
The present code was developed and tested with the following versions:

- Python 3.10
- numpy 1.24
- scipy 1.10
- matplotlib 3.7
- tensorflow 2.13
- tensorboard 2.13
- stable-baselines3 1.7.0


Make sure you have [Python3](https://www.python.org/downloads/) installed on
your computer.
You can then install the required packages by running
```bash
pip3 install -r requirements.txt
```
This will install all the needed packages which are listed in the requirements 
file. 
You can then run the training and testing by issuing
```bash
python3 main_training.py
python3 test.py
```


## Acknowledgements
This research was supported by the German Research Foundation (DFG) under grant
BE 8098/1-1, by the CELTIC-NEXT Project, 6G for Connected Sky (6G-SKY), with
funding received from Vinnova, Swedish Innovation Agency, and by the U.S.
National Science Foundation under Grants CNS-2128448 and ECCS-2335876.


## License and Referencing
This program is licensed under the GPLv3 license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

You can use the following BibTeX entry
```bibtex
@inproceedings{Meer2024icmlcn,
  author = {Meer, Irshad A. and Besser, Karl-Ludwig and Ozger, Mustafa and Schupke, Dominic and Poor, H. Vincent and Cavdar, Cicek},
  title = {Learning Based Dynamic Cluster Reconfiguration for UAV Mobility Management With 3D Beamforming},
  booktitle = {2024 IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN)},
  year = {2024},
  month = {5},
  publisher = {IEEE},
  venue = {Stockholm, Sweden},
  pages = {486--491},
  doi = {10.1109/ICMLCN59089.2024.10625071},
  archiveprefix = {arXiv},
  eprint = {2402.00224},
  primaryclass = {cs.IT},
}
```
