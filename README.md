# Self-supervised active learning with coreset based replay for Malware Detection
In this research, we propose semi-supervised active learning framework that integrates uncertainty-based coreset replay with self-supervised loss functions to achieve robust concept drift detection and adaptation in malware classification.

## Datasets

Download datasets [here](https://drive.google.com/file/d/1O0upEcTolGyyvasCPkZFY86FNclk29XO/view?usp=drive_link) from Google Drive. The zipped file contains DREBIN features of the APIGraph dataset and AndroZoo dataset we used.

Extract the downloaded file to `data/`, such that the datasets are under `data/gen_apigraph_drebin` and `data/gen_androzoo_drebin`.

* We collected `data/gen_apigraph_drebin` by downloading the sample hashes released by the APIGraph paper. The samples are from 2012 to 2018.
* We collected `data/gen_androzoo_drebin` by downloading apps from AndroZoo. The samples are from 2019 to 2021.

