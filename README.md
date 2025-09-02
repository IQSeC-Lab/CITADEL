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


## Requirements & Installation

After downloading the datasets and cloning the CITADEL repository, set up the environment using **Conda**:

```bash
conda env create -f env.yml
conda activate citadel-env
```

### Hardware Requirements
- CUDA-supported GPU (≥ 32 GB memory recommended)
- System RAM (≥ 32 GB)
We implemented and evaluated CITADEL on a dedicated research server equipped with:
- NVIDIA H100 NVL GPU
- 1.0 TB RAM
We also successfully ran experiments on **Google Colab Pro**

### Software Requirements
- Python 3.10
- PyTorch ≥ 2.0 with CUDA toolkit
- NumPy, Pandas, Matplotlib, SciPy
- scikit-learn, tqdm, tensorboard

All dependencies are specified in:
- environment.yml (Conda)

## Running CITADEL

CITADEL integrates:
- **Custom Augmentation**: e.g., `random_bit_flip_bernoulli`  
- **CITADEL Objective Function**: FixMatch + supervised contrastive loss  
- **Active Learning**: Multi-criteria sample selector (uncertainty, boundary, low-confident)  

To run **CITADEL** on all three Android malware benchmark datasets (**API-Graph, Chen-AndroZoo, LAMDA**) with a labeling budget of **400**, use:

```bash
./CITADEL/run_scripts.sh
```

#### Example: Running CITADEL with Active Learning on API-Graph

```bash
## Script to run CITADEL with active learning on the API-Graph dataset
DATASET="apigraph"
DATA_DIR="/home/mhaque3/myDir/data/gen_apigraph_drebin/"
UC='priority'
BUDGET=400
AUG="random_bit_flip_bernoulli"
SEED=220
LR=0.03
EPOCHS=200
RETRAIN_EPOCHS=70
BATCH_SIZE=512
AL_BATCH_SIZE=512

AL_START_YEAR=2013
AL_START_MONTH=7
AL_END_YEAR=2018
AL_END_MONTH=12

strategy="CITADEL"
LABEL_RATIO=0.4
LAMBDA_SUPCON=0.5
TS=$(date "+%m.%d-%H.%M.%S")

SAVE_PATH="CITADEL_results"
mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=1 nohup python -u CITADEL/citadel_fixmatch_al.py \
    --dataset $DATASET --data_dir $DATA_DIR \
    --lr $LR \
    --epochs $EPOCHS --retrain_epochs $RETRAIN_EPOCHS \
    --labeled_ratio $LABEL_RATIO \
    --lambda_supcon $LAMBDA_SUPCON \
    --batch_size $BATCH_SIZE \
    --al_batch_size $AL_BATCH_SIZE \
    --al_start_year $AL_START_YEAR --al_start_month $AL_START_MONTH \
    --al_end_year $AL_END_YEAR --al_end_month $AL_END_MONTH \
    --al \
    --supcon \
    --unc_samp $UC \
    --budget $BUDGET \
    --aug $AUG \
    --seed $SEED \
    --strategy $strategy \
    --save_path $SAVE_PATH > $SAVE_PATH/citadel_w_al_${DATASET}_${UC}_${AUG}_${BUDGET}_seed_${SEED}_${TS}.log 2>&1 &

```

Please make sure that we provide the correct dataset argument name with appropriate directory. The above command will also work on LAMDA dataset by changing the argument dataset name and directory. 

To run CITADEL on Chen-AndroZoo:
The **Chen-AndroZoo** dataset contains **16k+ features**, which increases memory requirements.  
To ensure stable training, we recommend reducing the **batch size** to `128` and the **learning rate** to `0.003`.


---

## Table II – CITADEL Performance (without Active Learning)

To reproduce **Table II** (baseline CITADEL without Active Learning), run the following command:

```bash
LABEL_RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
AUGS=("random_bit_flip_bernoulli" "random_bit_flip" "random_bit_flip_and_mask" "random_feature_mask")

DATASET="apigraph"   # change to "androzoo" or "lamda" as needed
DATA_DIR="/home/mhaque3/myDir/data/gen_apigraph_drebin/"
BATCH_SIZE=512
LR=0.03

mkdir -p CITADEL_table2

for AUG in "${AUGS[@]}"
do
    for RATIO in "${LABEL_RATIOS[@]}"
    do
        TS=$(date "+%m.%d-%H.%M.%S")
        LOGFILE="CITADEL_results/citadel_no_al_${DATASET}_${AUG}_ratio_${RATIO}_${TS}.log"
        echo "Running: Dataset=$DATASET | Aug=$AUG | Ratio=$RATIO"

        CUDA_VISIBLE_DEVICES=1 nohup python -u CITADEL/citadel_fixmatch_al.py \
            --dataset $DATASET \
            --data_dir $DATA_DIR \
            --lr $LR \
            --epochs 200 \
            --batch_size $BATCH_SIZE \
            --labeled_ratio $RATIO \
            --lambda_supcon 0.5 \
            --supcon \
            --aug $AUG \
            --seed 220 \
            --strategy CITADEL \
            --save_path CITADEL_table2 > $LOGFILE 2>&1 &
    done
done


# CUDA_VISIBLE_DEVICES=1 nohup python -u CITADEL/citadel_fixmatch_al.py \
#     --dataset <DATASET_NAME> \
#     --data_dir <DATASET_PATH> \
#     --lr <LEARNING_RATE> \
#     --epochs 200 \
#     --batch_size <BATCH_SIZE> \
#     --labeled_ratio 0.4 \
#     --lambda_supcon 0.5 \
#     --supcon \
#     --aug random_bit_flip_bernoulli \
#     --seed 220 \
#     --strategy CITADEL \
#     --save_path CITADEL_results > CITADEL_results/citadel_no_al_<DATASET_NAME>.log 2>&1 &
```

### Dataset-specific Configurations
#### APIGraph
```bash
--dataset apigraph
--data_dir /home/mhaque3/myDir/data/gen_apigraph_drebin/
--batch_size 512
--lr 0.03

```

### Key Arguments
| Argument           | Description                                                              |
| ------------------ | ------------------------------------------------------------------------ |
| `--dataset`        | Dataset name (`APIGraph`, `AndroZoo`, `LAMDA`)                           |
| `--data_dir`       | Path to dataset directory                                                |
| `--epochs`         | Number of training epochs                                                |
| `--retrain_epochs` | Number of retraining epochs for each active learning cycle               |
| `--batch_size`     | Training batch size                                                      |
| `--al_batch_size`  | Active learning retraining batch size                                    |
| `--lr`             | Learning rate                                                            |
| `--labeled_ratio`  | Fraction of labeled data used initially                                  |
| `--lambda_supcon`  | Weight for supervised contrastive loss                                   |
| `--al_start_year`  | Start year for testing (e.g., 2013)                                      |
| `--al_start_month` | Start month for testing (e.g., 7 for July)                               |
| `--al_end_year`    | End year for testing (e.g., 2018)                                        |
| `--al_end_month`   | End month for testing (e.g., 12 for December)                            |
| `--al`             | Enable Active Learning (default: False)                                  |
| `--supcon`         | Enable supervised contrastive loss                                       |
| `--strategy`       | Learning strategy (`CITADEL`, baseline variants)                         |
| `--unc_samp`       | Uncertainty sampling method (`lp-norm`, `boundary`, `priority`)          |
| `--budget`         | Number of samples to label per round in active learning                  |
| `--aug`            | Augmentation type (e.g., `random_bit_flip`, `random_bit_flip_bernoulli`) |
| `--seed`           | Random seed for reproducibility                                          |
| `--save_path`      | Path to save results and logs                                            |



## Acknowledgements

- We thank the authors of the paper "Continuous Learning for Android Malware Detection" and their GitHub repository [Chen-AL](https://github.com/wagner-group/active-learning) for providing the processed datasets (**APIGraph 2012–2018** and **AndroZoo 2019–2021**).  
- Thanks to the collaborators and my supervisor Dr. Mohammad Saidur Rahman for their guidance and support.  
