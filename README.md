<!-- # Self-supervised active learning with coreset based replay for Malware Detection

In this research, we propose semi-supervised active learning framework that integrates uncertainty-based coreset replay with self-supervised loss functions to achieve robust concept drift detection and adaptation in malware classification.

-->

# CITADEL: A Semi-Supervised Active Learning Framework for Malware Detection Under Continuous Distribution Drift

CITADEL is a semi-supervised active learning framework for Android malware detection designed to handle **long-term concept drift**.  
It integrates:
- Malware domain–focused binary feature augmentations
- FixMatch-based semi-supervised learning
- Prioritized multi-criteria active sample selection
- Enhanced objective function for separating boundary samples.
to adapt to malware concept drift.

The following figure shows the high-level pipeline of **CITADEL**:

<p align="center">
  <img src="CITADEL/figures/citadel_framework.png" alt="CITADEL Pipeline" width="600"/>
</p>

## Datasets

We evaluate on three Android malware datasets. Please follow the steps below to download and preprocess them:

We use the datasets provided in the paper "Continuous Learning for Android Malware Detection" and their GitHub repository: [active-learning](https://github.com/wagner-group/active-learning).  

### Download APIGraph (2012–2018) and Chen-AZ (2019-2021) dataset
- Download the datasets [here](https://drive.google.com/file/d/1O0upEcTolGyyvasCPkZFY86FNclk29XO/view?usp=drive_link) provided by the [this](https://github.com/wagner-group/active-learning) repository.
- Extract the downloaded zip into the `data/` directory:  `data/gen_apigraph_drebin` and `data/gen_androzoo_drebin`

### Download LAMDA (2013-2025) dataset
- use LAMDA official github repository [here](https://github.com/IQSeC-Lab/LAMDA) to download the dataset. You can also download the .npz format from [here](https://drive.google.com/drive/folders/19ysGjy5SU767lUBwc5-jLGTAmNas7W7P?usp=sharing).


## Run CITADEL
```
./CITADEL/run_scripts.sh
```


## Acknowledgements

- We thank the authors of the paper "Continuous Learning for Android Malware Detection" and their GitHub repository [Chen-AL](https://github.com/wagner-group/active-learning) for providing the processed datasets (**APIGraph 2012–2018** and **AndroZoo 2019–2021**).  
- Thanks to the collaborators and my supervisor Dr. Mohammad Saidur Rahman for their guidance and support.  
