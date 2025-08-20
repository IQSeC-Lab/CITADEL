# CUDA_VISIBLE_DEVICES=1 nohup python -u fixmatch_bit_flip_main.py > baseline_2runs/fixmatch_random_bit_flip_11_lbr_0.4_seed_1.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_bit_flip_main.py > baseline_2runs/fixmatch_random_bit_flip_11_lbr_0.4_seed_1.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python -u fixmatch_bit_flip_androzoo.py > baseline_2runs/fixmatch_androzoo_random_bit_flip_11_lbr_0.4_seed_1.log 2>&1 &
BUDGET=400
LP=1
AUG="random_bit_flip_bernoulli"
SEED=101
SAVE_PATH="uc_analysis"
# CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_al_uc.py --aug $AUG --seed $SEED --lp $LP --budget $BUDGET > logs_al_uc/fixmatch_w_al_${AUG}_11_lbr_0.4_bgt_${BUDGET}_lp_${LP}_seed_${SEED}.log 2>&1 &

# LP=2
# CUDA_VISIBLE_DEVICES=1 nohup python -u fixmatch_al_uc.py --aug $AUG --seed $SEED --lp $LP --budget $BUDGET > logs_al_uc/fixmatch_w_al_${AUG}_11_lbr_0.4_bgt_${BUDGET}_lp_${LP}_seed_${SEED}.log 2>&1 &

# LP=3
# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_al_uc.py --aug $AUG --seed $SEED --lp $LP --budget $BUDGET > logs_al_uc/fixmatch_w_al_${AUG}_11_lbr_0.4_bgt_${BUDGET}_lp_${LP}_seed_${SEED}.log 2>&1 &

# LP=9
# CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_al_uc.py --aug $AUG --seed $SEED --lp $LP --budget $BUDGET > logs_al_uc/fixmatch_w_al_${AUG}_11_lbr_0.4_bgt_${BUDGET}_lp_${LP}_seed_${SEED}.log 2>&1 &

# for LP in $(seq 1 20); do
#     echo "Running with aug=$AUG, LP=$LP, BUDGET=$BUDGET, seed=$SEED"
#     CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_al_uc_faster.py --lp $LP --budget $BUDGET --aug $AUG --seed $SEED --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_uc_${AUG}_${BUDGET}_lp_${LP}_seed_${SEED}.log 2>&1 &
#     wait  # Wait for all background jobs for this seed to finish
#     echo "Completed LP $LP"
# done

UC='boundary'
# for seed in $(seq 1 20); do
#     echo "Running with aug=$AUG, LP=$LP, BUDGET=$BUDGET, seed=$seed"
#     CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_al_uc_faster.py --unc_samp $UC --budget $BUDGET --aug $AUG --seed $seed --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_boundary_${AUG}_${BUDGET}_lp_${LP}_seed_${seed}.log 2>&1 &
#     wait  # Wait for all background jobs for this seed to finish
#     echo "Completed seed $seed"
# done


# LP=2
# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_al_uc_faster.py --lp $LP --budget $BUDGET --aug $AUG --seed $SEED --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_boundary_uc_${AUG}_${BUDGET}_lp_${LP}_seed_${SEED}.log 2>&1 &

# SAVE_PATH="analysis_boundary"
# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_al_uc_analysis.py --unc_samp $UC --budget $BUDGET --aug $AUG --seed $SEED --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_analysis_boundary_${AUG}_${BUDGET}_lp_${LP}_seed_${seed}.log 2>&1 &

BUDGET=50
AUG="random_bit_flip_bernoulli"
SEED=200
# SAVE_PATH="uc_analysis"
SAVE_PATH="analysis_boundary"
UC='boundary'
# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_al_uc_faster.py --unc_samp $UC --budget $BUDGET --aug $AUG --seed $SEED --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_boundary_uc_${AUG}_${BUDGET}_seed_${SEED}.log 2>&1 &




# BUDGET=400
# AUG="random_bit_flip_bernoulli"
# SEED=210
# # SAVE_PATH="uc_analysis"
# SAVE_PATH="analysis_boundary"
# UC='boundary'

# for seed in $(seq 100 110); do
#     echo "Running with aug=$AUG, BUDGET=$BUDGET, seed=$seed"
#     CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_al_uc_faster.py --unc_samp $UC --budget $BUDGET --aug $AUG --seed $seed --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_priority_${AUG}_${BUDGET}_seed_${seed}.log 2>&1 &
#     wait  # Wait for all background jobs for this seed to finish
#     echo "Completed seed $seed"
# done



# BUDGET=400
# AUG="random_bit_flip_bernoulli"
# SEED=220
# # SAVE_PATH="uc_analysis"
# SAVE_PATH="analysis_boundary"
# UC='boundary'
# strategy="unnorm_sup_con"
# # LABEL_RATIO=0.7
# LAMBDA_SUPCON=0.5
# # CUDA_VISIBLE_DEVICES=1 nohup python -u fixmatch_al_uc_faster.py --unc_samp $UC --budget $BUDGET --aug $AUG --seed $SEED --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_bernoulli_0.1_${UC}_${AUG}_${BUDGET}_seed_${SEED}.log 2>&1 &
# for seed in $(seq 221 221); do
#     CUDA_VISIBLE_DEVICES=1 nohup python -u fixmatch_al_uc_triplet.py --unc_samp $UC --budget $BUDGET \
#         --aug $AUG --lambda_supcon $LAMBDA_SUPCON --seed $seed --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${LAMBDA_SUPCON}_${UC}_${AUG}_${BUDGET}_seed_${seed}.log 2>&1 &
#     # wait
# done



# BUDGET=400
# AUG="random_bit_flip_bernoulli"
# SEED=220
# # SAVE_PATH="uc_analysis"
# SAVE_PATH="analysis_boundary"
# UC='boundary'
# strategy="self_training"
# # LABEL_RATIO=0.7
# LAMBDA_SUPCON=0.5
# # CUDA_VISIBLE_DEVICES=1 nohup python -u fixmatch_al_uc_faster.py --unc_samp $UC --budget $BUDGET --aug $AUG --seed $SEED --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_bernoulli_0.1_${UC}_${AUG}_${BUDGET}_seed_${SEED}.log 2>&1 &
# for seed in $(seq 221 221); do
#     CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_self_training.py --unc_samp $UC --budget $BUDGET \
#         --aug $AUG --lambda_supcon $LAMBDA_SUPCON --seed $seed --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${LAMBDA_SUPCON}_${UC}_${AUG}_${BUDGET}_seed_${seed}.log 2>&1 &
#     # wait
# done




# BUDGET=400
# AUG="random_bit_flip_bernoulli"
# SEED=420
# # SAVE_PATH="uc_analysis"
# SAVE_PATH="al_lp_api"
# UC='lp-norm'
# strategy="lp"
# # LABEL_RATIO=0.7
# LAMBDA_SUPCON=0.5

# for lp in $(seq 1.0 0.2 5.0); do
#     echo "Running with aug=$AUG, BUDGET=$BUDGET, seed=$SEED"
#     CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_al_uc_supcon.py --al --unc_samp $UC --lp $lp --budget $BUDGET \
#         --aug $AUG --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${UC}_${lp}_${AUG}_${BUDGET}_seed_${SEED}.log 2>&1 &
#     wait
# done


BUDGET=400
AUG="random_bit_flip_bernoulli"
SEED=720
# SAVE_PATH="uc_analysis"
SAVE_PATH="t_sne_retrain_feature"
UC='boundary'
strategy="lamda_supcon"
# LABEL_RATIO=0.7
LAMBDA_SUPCON=0.5
TS=$(date "+%m.%d-%H.%M.%S")
# echo "Running self-training for AndroZoo dataset"
# CUDA_VISIBLE_DEVICES=1 python -u fixmatch_al_uc_analysis.py --unc_samp $UC --budget $BUDGET --save_path $SAVE_PATH --aug $AUG > $SAVE_PATH/fixmatch_w_al_${strategy}.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_al_uc_androzoo.py --al --self_training --unc_samp $UC --budget $BUDGET \
#         --aug $AUG --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${UC}_${lp}_${AUG}_${BUDGET}_seed_${SEED}.log 2>&1 &

UC='priority'
SAVE_PATH="al_lp_lamda"
CUDA_VISIBLE_DEVICES=1 nohup python -u fixmatch_al_uc_lamda.py --al --supcon --unc_samp $UC --budget $BUDGET \
        --aug $AUG --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${UC}_${AUG}_${BUDGET}_seed_${SEED}_${TS}.log 2>&1 &
    
# for lp in $(seq 5.0 0.5 9.0); do
#     echo "Running with aug=$AUG, BUDGET=$BUDGET, seed=$SEED"
#     CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_al_uc_lamda.py --al --unc_samp $UC --lp $lp --budget $BUDGET \
#         --aug $AUG --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${UC}_${lp}_${AUG}_${BUDGET}_seed_${SEED}.log 2>&1 &
#     wait
# done




# BUDGETS=(50 100 200 400)
# AUG="random_bit_flip_bernoulli"
# SEED=222
# # SAVE_PATH="uc_analysis"
# SAVE_PATH="/home/mhaque3/myDir/SSL-malware/baseline_experiments/baseline_api_multi_label_ratio"
# strategy="supcon_inc_boundary"
# # LABEL_RATIOS=(0.4 0.5 0.6 0.7 0.8 0.9)
# LAMBDA_SUPCON=0.5
# # AUGS=("random_bit_flip" "random_bit_flip_bernoulli" "random_bit_flip_and_mask" "random_feature_mask")

# strategy="baseline"
# ratio=0.4
# for SEED in $(seq 300 302)
# do
#     for BUDGET in "${BUDGETS[@]}"
#     do
#         strategy="final"
#         SAVE_PATH="/home/mhaque3/myDir/SSL-malware/baseline_experiments/al_apigraph"
#         echo "Running with aug=$AUG, BUDGET=$BUDGET, ratio=$ratio, seed=$SEED"
#         UC='boundary'
#         CUDA_VISIBLE_DEVICES=1 nohup python -u fixmatch_al_uc_supcon.py --unc_samp $UC --budget $BUDGET \
#             --aug $AUG --al --supcon --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}.log 2>&1 &
        

#         # SEED=222
#         SAVE_PATH="/home/mhaque3/myDir/SSL-malware/baseline_experiments/al_androzoo"
#         echo "Running with aug=$AUG, BUDGET=$BUDGET, ratio=$ratio, seed=$SEED"
#         UC='boundary'
#         CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_al_uc_androzoo.py --unc_samp $UC --budget $BUDGET \
#             --aug $AUG --al --supcon --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}.log 2>&1 &
        
#         # SEED=400
#         SAVE_PATH="/home/mhaque3/myDir/SSL-malware/baseline_experiments/al_LAMDA"
#         echo "Running with aug=$AUG, BUDGET=$BUDGET, ratio=$ratio, seed=$SEED"
#         UC='boundary'
#         CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_al_uc_lamda.py --unc_samp $UC --budget $BUDGET \
#             --aug $AUG --al --supcon --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}.log 2>&1 &
        
#         # SEED=301
#         # UC='boundary'
#         # CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_al_uc_supcon.py --unc_samp $UC --budget $BUDGET \
#         #     --aug $AUG --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${LAMBDA_SUPCON}_${UC}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}.log 2>&1 &
#         # SEED=224
#         # UC='priority'
#         # CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_al_uc_supcon.py --unc_samp $UC --budget $BUDGET \
#         #     --aug $AUG --al --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${LAMBDA_SUPCON}_${UC}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}.log 2>&1 &
#         # SEED=225
#         # UC='priority'
#         # CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_al_uc_supcon.py --unc_samp $UC --budget $BUDGET \
#         #     --aug $AUG --al --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${LAMBDA_SUPCON}_${UC}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}.log 2>&1 &
        
#         wait
#         echo "Completed ratio $BUDGET"
#     done
# done




# TIME=$(date "+%m.%d-%H.%M.%S")
# BUDGET=400
# AUG="random_bit_flip_bernoulli"
# SEED=220
# # UC='boundary'
# strategy="profiler"
# ratio=0.7
# LAMBDA_SUPCON=0.5
# DATA=/home/mhaque3/myDir/data/gen_apigraph_drebin_multi/500000/
# strategy="profiler_500000"
# SAVE_PATH="/home/mhaque3/myDir/SSL-malware/baseline_experiments/profiler_results"
# echo "Running with aug=$AUG, BUDGET=$BUDGET, ratio=$ratio, seed=$SEED"
# UC='boundary'
# CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_al_uc_profiler.py --data_path $DATA --unc_samp $UC --budget $BUDGET \
#     --aug $AUG --al --supcon --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_new_${strategy}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}_${TIME}.log 2>&1



# TIME=$(date "+%m.%d-%H.%M.%S")
# BUDGET=400
# AUG="random_bit_flip_bernoulli"
# SEED=220
# # UC='boundary'
# strategy="profiler"
# ratio=0.7
# LAMBDA_SUPCON=0.5
# DATA=/home/mhaque3/myDir/data/gen_apigraph_drebin_multi/1000000/
# strategy="profiler_1000000"
# SAVE_PATH="/home/mhaque3/myDir/SSL-malware/baseline_experiments/profiler_results"
# echo "Running with aug=$AUG, BUDGET=$BUDGET, ratio=$ratio, seed=$SEED"
# UC='boundary'
# CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_al_uc_profiler.py --unc_samp $UC --budget $BUDGET \
#     --aug $AUG --al --supcon --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_new_${strategy}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}_${TIME}.log 2>&1


# TIME=$(date "+%m.%d-%H.%M.%S")
# BUDGET=400
# AUG="random_bit_flip_bernoulli"
# SEED=220
# UC='boundary'

# ratio=0.7
# LAMBDA_SUPCON=0.5
# DATA=/home/mhaque3/myDir/data/gen_apigraph_drebin_multi/1000/
# strategy="profiler_1000"
# SAVE_PATH="/home/mhaque3/myDir/SSL-malware/baseline_experiments/profiler_results"
# echo "Running with aug=$AUG, BUDGET=$BUDGET, ratio=$ratio, seed=$SEED"
# UC='boundary'
# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_al_uc_profiler.py --data_path $DATA --unc_samp $UC --budget $BUDGET \
#     --aug $AUG --al --supcon --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_new_${strategy}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}_${TIME}.log 2>&1



# TIME=$(date "+%m.%d-%H.%M.%S")
# BUDGET=400
# AUG="random_bit_flip_bernoulli"
# SEED=220
# # UC='boundary'
# strategy="profiler"
# ratio=0.7
# LAMBDA_SUPCON=0.5
# DATA=/home/mhaque3/myDir/data/gen_apigraph_drebin_multi/5000/
# strategy="profiler_5000"
# SAVE_PATH="/home/mhaque3/myDir/SSL-malware/baseline_experiments/profiler_results"
# echo "Running with aug=$AUG, BUDGET=$BUDGET, ratio=$ratio, seed=$SEED"
# UC='boundary'
# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_al_uc_profiler.py --data_path $DATA --unc_samp $UC --budget $BUDGET \
#     --aug $AUG --al --supcon --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_new_${strategy}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}_${TIME}.log 2>&1




# TIME=$(date "+%m.%d-%H.%M.%S")
# BUDGET=400
# AUG="random_bit_flip_bernoulli"
# SEED=220
# # UC='boundary'
# strategy="profiler"
# ratio=0.7
# LAMBDA_SUPCON=0.5
# DATA=/home/mhaque3/myDir/data/gen_apigraph_drebin_multi/10000/
# strategy="profiler_10000"
# SAVE_PATH="/home/mhaque3/myDir/SSL-malware/baseline_experiments/profiler_results"
# echo "Running with aug=$AUG, BUDGET=$BUDGET, ratio=$ratio, seed=$SEED"
# UC='boundary'
# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_al_uc_profiler.py --data_path $DATA --unc_samp $UC --budget $BUDGET \
#     --aug $AUG --al --supcon --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_new_${strategy}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}_${TIME}.log 2>&1



# TIME=$(date "+%m.%d-%H.%M.%S")
# BUDGET=400
# AUG="random_bit_flip_bernoulli"
# SEED=220
# # UC='boundary'
# strategy="profiler"
# ratio=0.7
# LAMBDA_SUPCON=0.5
# DATA=/home/mhaque3/myDir/data/gen_apigraph_drebin_multi/20000/
# strategy="profiler_20000"
# SAVE_PATH="/home/mhaque3/myDir/SSL-malware/baseline_experiments/profiler_results"
# echo "Running with aug=$AUG, BUDGET=$BUDGET, ratio=$ratio, seed=$SEED"
# UC='boundary'
# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_al_uc_profiler.py --data_path $DATA --unc_samp $UC --budget $BUDGET \
#     --aug $AUG --al --supcon --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_new_${strategy}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}_${TIME}.log 2>&1



# TIME=$(date "+%m.%d-%H.%M.%S")
# BUDGET=400
# AUG="random_bit_flip_bernoulli"
# SEED=220
# # UC='boundary'
# strategy="profiler"
# ratio=0.7
# LAMBDA_SUPCON=0.5
# DATA=/home/mhaque3/myDir/data/gen_apigraph_drebin_multi/30000/
# strategy="profiler_30000"
# SAVE_PATH="/home/mhaque3/myDir/SSL-malware/baseline_experiments/profiler_results"
# echo "Running with aug=$AUG, BUDGET=$BUDGET, ratio=$ratio, seed=$SEED"
# UC='boundary'
# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_al_uc_profiler.py --data_path $DATA --unc_samp $UC --budget $BUDGET \
#     --aug $AUG --al --supcon --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_new_${strategy}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}_${TIME}.log 2>&1



# TIME=$(date "+%m.%d-%H.%M.%S")
# BUDGET=400
# AUG="random_bit_flip_bernoulli"
# SEED=220
# # UC='boundary'
# strategy="profiler"
# ratio=0.7
# LAMBDA_SUPCON=0.5
# DATA=/home/mhaque3/myDir/data/gen_apigraph_drebin_multi/50000/
# strategy="profiler_50000"
# SAVE_PATH="/home/mhaque3/myDir/SSL-malware/baseline_experiments/profiler_results"
# echo "Running with aug=$AUG, BUDGET=$BUDGET, ratio=$ratio, seed=$SEED"
# UC='boundary'
# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_al_uc_profiler.py --data_path $DATA --unc_samp $UC --budget $BUDGET \
#     --aug $AUG --al --supcon --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_new_${strategy}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}_${TIME}.log 2>&1








# BUDGET=400
# AUG="random_bit_flip_bernoulli"
# SEED=222
# # SAVE_PATH="uc_analysis"
# SAVE_PATH="/home/mhaque3/myDir/SSL-malware/baseline_experiments/baseline_api_multi_label_ratio"
# strategy="supcon_inc_boundary"
# LABEL_RATIOS=(0.4 0.5 0.6 0.7 0.8 0.9)
# LAMBDA_SUPCON=0.5
# AUGS=("random_bit_flip" "random_bit_flip_bernoulli" "random_bit_flip_and_mask" "random_feature_mask")

# strategy="baseline"

# for AUG in "${AUGS[@]}"
# do
#     for ratio in "${LABEL_RATIOS[@]}"
#     do
#         SEED=300
#         SAVE_PATH="/home/mhaque3/myDir/SSL-malware/baseline_experiments/baseline_api_multi_label_ratio"
#         echo "Running with aug=$AUG, BUDGET=$BUDGET, ratio=$ratio, seed=$SEED"
#         UC='boundary'
#         CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_al_uc_supcon.py --unc_samp $UC --budget $BUDGET \
#             --aug $AUG --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}.log 2>&1 &
        

#         SEED=222
#         SAVE_PATH="/home/mhaque3/myDir/SSL-malware/baseline_experiments/baseline_andro_multi_label_ratio"
#         echo "Running with aug=$AUG, BUDGET=$BUDGET, ratio=$ratio, seed=$SEED"
#         UC='boundary'
#         CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_al_uc_androzoo.py --unc_samp $UC --budget $BUDGET \
#             --aug $AUG --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}.log 2>&1 &
        
#         SEED=400
#         SAVE_PATH="/home/mhaque3/myDir/SSL-malware/baseline_experiments/baseline_LAMDA_multi_label_ratio"
#         echo "Running with aug=$AUG, BUDGET=$BUDGET, ratio=$ratio, seed=$SEED"
#         UC='boundary'
#         CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_al_uc_lamda.py --unc_samp $UC --budget $BUDGET \
#             --aug $AUG --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}.log 2>&1 &
        
#         # SEED=301
#         # UC='boundary'
#         # CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_al_uc_supcon.py --unc_samp $UC --budget $BUDGET \
#         #     --aug $AUG --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${LAMBDA_SUPCON}_${UC}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}.log 2>&1 &
#         # SEED=224
#         # UC='priority'
#         # CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_al_uc_supcon.py --unc_samp $UC --budget $BUDGET \
#         #     --aug $AUG --al --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${LAMBDA_SUPCON}_${UC}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}.log 2>&1 &
#         # SEED=225
#         # UC='priority'
#         # CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_al_uc_supcon.py --unc_samp $UC --budget $BUDGET \
#         #     --aug $AUG --al --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${LAMBDA_SUPCON}_${UC}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}.log 2>&1 &
        
#         wait
#         echo "Completed ratio $ratio"
#     done
# done




# # baseline run for Androzoo dataset
# BUDGET=400
# AUG="random_bit_flip_bernoulli"
# SEED=222
# # SAVE_PATH="uc_analysis"
# SAVE_PATH="/home/mhaque3/myDir/SSL-malware/baseline_experiments/baseline_andro_original"
# strategy="supcon_inc_boundary"
# LABEL_RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# LAMBDA_SUPCON=0.5
# AUGS=("random_bit_flip" "random_bit_flip_bernoulli" "random_bit_flip_and_mask" "random_feature_mask")

# strategy="baseline"

# for AUG in "${AUGS[@]}"
# do
#     for ratio in "${LABEL_RATIOS[@]}"
#     do
#         SEED=222
#         echo "Running with aug=$AUG, BUDGET=$BUDGET, ratio=$ratio, seed=$SEED"
#         UC='boundary'
#         CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_al_uc_androzoo.py --aug $AUG --labeled_ratio $ratio\
#          --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}.log 2>&1 &
        
#         # SEED=300
#         # UC='boundary'
#         # CUDA_VISIBLE_DEVICES=1 nohup python -u fixmatch_al_uc_supcon.py --unc_samp $UC --budget $BUDGET \
#         #     --aug $AUG --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${LAMBDA_SUPCON}_${UC}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}.log 2>&1 &
#         # SEED=224
#         # UC='priority'
#         # CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_al_uc_supcon.py --unc_samp $UC --budget $BUDGET \
#         #     --aug $AUG --al --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${LAMBDA_SUPCON}_${UC}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}.log 2>&1 &
#         # SEED=225
#         # UC='priority'
#         # CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_al_uc_supcon.py --unc_samp $UC --budget $BUDGET \
#         #     --aug $AUG --al --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${LAMBDA_SUPCON}_${UC}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}.log 2>&1 &
        
#         wait
#         echo "Completed ratio $ratio"
#     done
# done





# ## For baseline lamda 
# ## baseline run for Androzoo dataset
# BUDGET=400
# AUG="random_bit_flip_bernoulli"
# SEED=222
# # SAVE_PATH="uc_analysis"
# SAVE_PATH="/home/mhaque3/myDir/SSL-malware/baseline_experiments/baseline_LAMDA_multi_label_ratio"
# strategy="supcon_inc_boundary"
# LABEL_RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# LAMBDA_SUPCON=0.5
# AUGS=("random_bit_flip" "random_bit_flip_bernoulli" "random_bit_flip_and_mask" "random_feature_mask")

# strategy="baseline"

# for AUG in "${AUGS[@]}"
# do
#     for ratio in "${LABEL_RATIOS[@]}"
#     do
#         SEED=400
#         echo "Running with aug=$AUG, BUDGET=$BUDGET, ratio=$ratio, seed=$SEED"
#         UC='boundary'
#         CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_al_uc_lamda.py --unc_samp $UC --budget $BUDGET \
#             --aug $AUG --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}.log 2>&1 &
        
#         # SEED=401
#         # UC='boundary'
#         # CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_al_uc_supcon.py --unc_samp $UC --budget $BUDGET \
#         #     --aug $AUG --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${LAMBDA_SUPCON}_${UC}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}.log 2>&1 &
#         # SEED=224
#         # UC='priority'
#         # CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_al_uc_supcon.py --unc_samp $UC --budget $BUDGET \
#         #     --aug $AUG --al --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${LAMBDA_SUPCON}_${UC}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}.log 2>&1 &
#         # SEED=225
#         # UC='priority'
#         # CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_al_uc_supcon.py --unc_samp $UC --budget $BUDGET \
#         #     --aug $AUG --al --labeled_ratio $ratio --seed $SEED --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${LAMBDA_SUPCON}_${UC}_${AUG}_${BUDGET}_${ratio}_seed_${SEED}.log 2>&1 &
        
#         wait
#         echo "Completed ratio $ratio"
#     done
# done




# BUDGET=400
# AUG="random_bit_flip_bernoulli"
# SEED=220
# # SAVE_PATH="uc_analysis"
# SAVE_PATH="lamda_output"
# UC='priority'
# strategy="wo_supcon_lamda_cos_scheduler"
# # LABEL_RATIO=0.7
# LAMBDA_SUPCON=0.5

# for seed in $(seq 220 220); do
#     CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_al_uc_lamda.py --unc_samp $UC --budget $BUDGET \
#         --aug $AUG --seed $seed --strategy $strategy --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_${strategy}_${LAMBDA_SUPCON}_${UC}_${AUG}_${BUDGET}_seed_${seed}.log 2>&1 &
#     # wait
# done



# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_al_uc_faster.py --unc_samp $UC --budget $BUDGET --aug $AUG --seed $SEED --save_path $SAVE_PATH > $SAVE_PATH/fixmatch_w_al_priority_uc_${AUG}_${BUDGET}_seed_${SEED}.log 2>&1 &



# SEED=10
# for LP in $(seq 2 30); do
#     echo "Running with aug=$AUG, LP=$LP, BUDGET=$BUDGET, labeled_ratio=0.4, seed=$SEED"
#     CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_al_uc_faster.py --aug $AUG --seed $SEED --lp $LP --budget $BUDGET > logs_al_uc/fixmatch_faster_w_al_${AUG}_11_lbr_0.4_bgt_${BUDGET}_lp_${LP}_seed_${SEED}.log 2>&1 &
#     wait  # Wait for all background jobs for this seed to finish
#     echo "Completed seed $SEED"
# done




# !/bin/bash

# AUGS=("random_bit_flip")
# RATIO=0.4
# N_BIT=11

# # You can set seeds explicitly, or use sequence
# for SEED in $(seq 1 20); do
#   for AUG in "${AUGS[@]}"; do
#     echo "Running with aug=$AUG, bit_flip=$N_BIT, labeled_ratio=$RATIO, seed=$SEED"
    
#     CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_bit_flip_main.py \
#       --aug $AUG \
#       --bit_flip $N_BIT \
#       --labeled_ratio $RATIO \
#       --seed $SEED \
#       > baseline_mSeed/fixmatch_${AUG}_${N_BIT}_lbr_${RATIO}_seed_${SEED}.log 2>&1 &
    
#   done
#   wait  # Wait for all background jobs for this seed to finish
# done





#!/bin/bash

# AUGS=("random_bit_flip_bernoulli" "random_bit_flip")
# RATIO=0.4
# N_BIT=11

# # You can set seeds explicitly, or use sequence
# for SEED in $(seq 100 200); do
#   for AUG in "${AUGS[@]}"; do
#     echo "Running with aug=$AUG, bit_flip=$N_BIT, labeled_ratio=$RATIO, seed=$SEED"
#     # run for aug 1 in gpu 2
#     if [[ $AUG == "random_bit_flip" ]]; then
#         CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_bit_flip_main.py \
#         --aug $AUG \
#         --bit_flip $N_BIT \
#         --labeled_ratio $RATIO \
#         --seed $SEED \
#         > baseline_mSeed/fixmatch_${AUG}_${N_BIT}_lbr_${RATIO}_seed_${SEED}.log 2>&1 &
#     fi
#     # run for aug 2 in different gpu
#     if [[ $AUG == "random_bit_flip_bernoulli" ]]; then
#       CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_bit_flip_main.py \
#         --aug $AUG \
#         --bit_flip $N_BIT \
#         --labeled_ratio $RATIO \
#         --seed $SEED \
#         > baseline_mSeed/fixmatch_${AUG}_${N_BIT}_lbr_${RATIO}_seed_${SEED}.log 2>&1 &
#     fi
#   done
#   wait  # Wait for all background jobs for this seed to finish
# done







#!/bin/bash

# AUGS=("random_bit_flip" "random_bit_flip_bernoulli" "random_bit_flip_and_mask" "random_feature_mask")
# LABEL_RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# SEEDS=(0 1)

# for n_bit in {6..30}
# do
#   for aug in "${AUGS[@]}"
#   do
#     for ratio in "${LABEL_RATIOS[@]}"
#     do
#       for seed in "${SEEDS[@]}"
#       do
#         # # check if n_bit is eq 4 and aug is random_bit_flip or aug is random_bit_flip_bernoulli
#         # if [[ $n_bit -eq 4 && ( $aug == "random_bit_flip" || $aug == "random_bit_flip_bernoulli" || $aug == "random_bit_flip_and_mask") ]]; then
#         #   continue  # Skip this iteration
#         # fi
  
#         echo "Running with aug=$aug, bit_flip=$n_bit, labeled_ratio=$ratio, seed=$seed"
#         CUDA_VISIBLE_DEVICES=1 nohup python -u fixmatch_bit_flip_main.py \
#           --aug $aug \
#           --bit_flip $n_bit \
#           --labeled_ratio $ratio \
#           --seed $seed \
#           > baseline_logs/fixmatch_${aug}_${n_bit}_lbr_${ratio}_seed_${seed}.log 2>&1 &
#       done
#       wait
#     done
#   done
# done
# wait




# AUGS=("random_bit_flip")
# # LABEL_RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# ratio=0.4
# SEEDS=(0 1)
# n_bit=11

# for i in {1..20}
# do
#   for aug in "${AUGS[@]}"
#   do
#     seed=$i
#     echo "Running with aug=$aug, bit_flip=$n_bit, labeled_ratio=$ratio, seed=$seed"
#     CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_bit_flip_main.py \
#       --aug $aug \
#       --bit_flip $n_bit \
#       --labeled_ratio $ratio \
#       --seed $seed \
#       > baseline_mSeed/fixmatch_${aug}_${n_bit}_lbr_${ratio}_seed_${seed}.log 2>&1 &
    
#     # seed=2
#     # echo "Running with aug=$aug, bit_flip=$n_bit, labeled_ratio=$ratio, seed=$seed"
#     # CUDA_VISIBLE_DEVICES=1 nohup python -u fixmatch_bit_flip_main.py \
#     #   --aug $aug \
#     #   --bit_flip $n_bit \
#     #   --labeled_ratio $ratio \
#     #   --seed $seed \
#     #   > baseline_2runs/fixmatch_${aug}_${n_bit}_lbr_${ratio}_seed_${seed}.log 2>&1 &
    
#     # seed=3
#     # echo "Running with aug=$aug, bit_flip=$n_bit, labeled_ratio=$ratio, seed=$seed"
#     # CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_bit_flip_main.py \
#     #   --aug $aug \
#     #   --bit_flip $n_bit \
#     #   --labeled_ratio $ratio \
#     #   --seed $seed \
#     #   > baseline_2runs/fixmatch_${aug}_${n_bit}_lbr_${ratio}_seed_${seed}.log 2>&1 &
    
#     # seed=4
#     # echo "Running with aug=$aug, bit_flip=$n_bit, labeled_ratio=$ratio, seed=$seed"
#     # CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_bit_flip_main.py \
#     #   --aug $aug \
#     #   --bit_flip $n_bit \
#     #   --labeled_ratio $ratio \
#     #   --seed $seed \
#     #   > baseline_2runs/fixmatch_${aug}_${n_bit}_lbr_${ratio}_seed_${seed}.log 2>&1 &
#     wait
#   done
# done






# AUGS=("random_bit_flip" "random_bit_flip_bernoulli")
# # LABEL_RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# ratio=0.4
# SEEDS=(0 1)
# n_bit=11

# for aug in "${AUGS[@]}"
# do
#   seed=1
#   echo "Running with aug=$aug, bit_flip=$n_bit, labeled_ratio=$ratio, seed=$seed"
#   CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_bit_flip_main.py \
#     --aug $aug \
#     --bit_flip $n_bit \
#     --labeled_ratio $ratio \
#     --seed $seed \
#     > baseline_2runs/fixmatch_${aug}_${n_bit}_lbr_${ratio}_seed_${seed}.log 2>&1 &
  
#   seed=2
#   echo "Running with aug=$aug, bit_flip=$n_bit, labeled_ratio=$ratio, seed=$seed"
#   CUDA_VISIBLE_DEVICES=1 nohup python -u fixmatch_bit_flip_main.py \
#     --aug $aug \
#     --bit_flip $n_bit \
#     --labeled_ratio $ratio \
#     --seed $seed \
#     > baseline_2runs/fixmatch_${aug}_${n_bit}_lbr_${ratio}_seed_${seed}.log 2>&1 &
  
#   seed=3
#   echo "Running with aug=$aug, bit_flip=$n_bit, labeled_ratio=$ratio, seed=$seed"
#   CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_bit_flip_main.py \
#     --aug $aug \
#     --bit_flip $n_bit \
#     --labeled_ratio $ratio \
#     --seed $seed \
#     > baseline_2runs/fixmatch_${aug}_${n_bit}_lbr_${ratio}_seed_${seed}.log 2>&1 &
  
#   seed=4
#   echo "Running with aug=$aug, bit_flip=$n_bit, labeled_ratio=$ratio, seed=$seed"
#   CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_bit_flip_main.py \
#     --aug $aug \
#     --bit_flip $n_bit \
#     --labeled_ratio $ratio \
#     --seed $seed \
#     > baseline_2runs/fixmatch_${aug}_${n_bit}_lbr_${ratio}_seed_${seed}.log 2>&1 &
#   wait
# done




#!/bin/bash

# AUGS=("random_bit_flip" "random_bit_flip_bernoulli")
# LABEL_RATIO=0.4
# N_BITS_RANGE=({11..12})

# # Define the seeds and their corresponding GPUs in arrays
# SEEDS=(42 62 72 82)
# GPUS=(0 1 2 3)

# # Create the results directory
# RESULTS_DIR="baseline_2runs"

# # --- Main Experiment Loop ---
# for n_bit in "${N_BITS_RANGE[@]}"; do
#   for aug in "${AUGS[@]}"; do
    
#     echo "Starting parallel batch for: aug=$aug, n_bit=$n_bit"

#     # Loop through the indices of the SEEDS array to pair them with GPUS
#     for i in "${!SEEDS[@]}"; do
#       # Get the specific seed and GPU for this run
#       seed=${SEEDS[$i]}
#       gpu=${GPUS[i]}

#       LOG_FILE="${RESULTS_DIR}/fixmatch_${aug}_${n_bit}_lbr_${LABEL_RATIO}_seed_${seed}.log"

#       echo "--> Submitting job on GPU $gpu with Seed: $seed. Log: $LOG_FILE"

#       # This single command block replaces all the repeated code
#       CUDA_VISIBLE_DEVICES=$gpu nohup python -u fixmatch_wo_al.py \
#         --aug "$aug" \
#         --bit_flip "$n_bit" \
#         --labeled_ratio "$LABEL_RATIO" \
#         --seed "$seed" \
#         > "$LOG_FILE" 2>&1 &
#     done

#     # 'wait' pauses the script until ALL 4 background jobs in this batch finish
#     echo "Waiting for batch (aug=$aug, n_bit=$n_bit) to complete..."
#     wait
#     echo "Batch finished."
    
#   done
# done

# echo "All experiments have completed."




# AUGS=("random_bit_flip" "random_bit_flip_bernoulli" "random_bit_flip_and_mask", "random_feature_mask")


# for n_bit in {4..30}
# do
#   for aug in "${AUGS[@]}"
#   do  
#     echo "Running with aug=$aug, bit_flip=$n_bit"
#     CUDA_VISIBLE_DEVICES=1 nohup python -u fixmatch_bit_flip_main.py --aug $aug --bit_flip $n_bit > logs/fixmatch_${aug}_${n_bit}.log 2>&1
#   done
#   # wait  # Wait for all n_bit jobs for this aug to finish
# done

# for n_bit in {8..9}
# do
#   CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_bit_flip_main.py --aug random_feature_mask --bit_flip $n_bit > logs_feature_mask/fixmatch_random_feature_mask_${n_bit}.log 2>&1
# done

# NOTE: using active learning with FixMatch on malware dataset
# seed=60
# CUDA_VISIBLE_DEVICES=1 nohup python -u fixmatch_bit_flip_w_al.py --aug "random_bit_flip_bernoulli" --seed $seed > logs_al/fixmatch_al_bits_flip_bernoulli_0.05_lbr_0.4_s_${seed}.log 2>&1 &
# seed=20
# CUDA_VISIBLE_DEVICES=1 nohup python -u fixmatch_bit_flip_w_al.py --seed $seed > logs_al/fixmatch_aug_default_bits_flip_11_lbr_0.4_s_${seed}.log 2>&1 &
# seed=30
# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_bit_flip_w_al.py --seed $seed > logs_al/fixmatch_aug_default_bits_flip_11_lbr_0.4_s_${seed}.log 2>&1 &
# seed=40
# CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_bit_flip_w_al.py --seed $seed > logs_al/fixmatch_aug_default_bits_flip_11_lbr_0.4_s_${seed}.log 2>&1 &


# seed=10
# CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_al_analysis.py --seed $seed > logs_al_uc/fixmatch_al_uc_bits_flip_11_lbr_0.4_s_${seed}.log 2>&1 &



# # The following script is used to run the FixMatch model with different labeled ratios on a malware dataset.
# CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_bit_flip_main.py > fixmatch_bit_flip_main_sgd_cosine_step_seed_0.4.log 2>&1 &



# CUDA_VISIBLE_DEVICES=1 nohup python -u fixmatch_bit_flip_main.py --labeled_ratio .2 > fixmatch_bit_flip_main_.2.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python -u fixmatch_bit_flip_main.py --labeled_ratio .3 > fixmatch_bit_flip_main_.3.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_bit_flip_main.py --labeled_ratio .4 > fixmatch_bit_flip_main_.4.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_bit_flip_main.py --labeled_ratio .5 > fixmatch_bit_flip_main_.5.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_bit_flip_main.py --labeled_ratio .6 > fixmatch_bit_flip_main_.6.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_bit_flip_main.py --labeled_ratio .7 > fixmatch_bit_flip_main_.7.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_bit_flip_main.py --labeled_ratio .8 > fixmatch_bit_flip_main_.8.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_bit_flip_main.py --labeled_ratio .9 > fixmatch_bit_flip_main_.9.log 2>&1 &





# The following script is used to run the FixMatch model with bit flip experiments on a malware dataset.

# CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_bit_flip_main.py --bit_flip 3 > fixmatch_bit_flip_main_3.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_bit_flip_main.py --bit_flip 4 > fixmatch_bit_flip_main_4.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_bit_flip_main.py --bit_flip 5 > fixmatch_bit_flip_main_5.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_bit_flip_main.py --bit_flip 6 > fixmatch_bit_flip_main_6.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_bit_flip_main.py --bit_flip 7 > fixmatch_bit_flip_main_7.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_bit_flip_main.py --bit_flip 8 > fixmatch_bit_flip_main_8.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_bit_flip_main.py --bit_flip 9 > fixmatch_bit_flip_main_9.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_bit_flip_main.py --bit_flip 10 > fixmatch_bit_flip_main_10.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_bit_flip_main.py --bit_flip 11 > fixmatch_bit_flip_main_11.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_bit_flip_main.py --bit_flip 12 > fixmatch_bit_flip_main_12.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_bit_flip_main.py --bit_flip 13 > fixmatch_bit_flip_main_13.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_bit_flip_main.py --bit_flip 14 > fixmatch_bit_flip_main_14.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_bit_flip_main.py --bit_flip 15 > fixmatch_bit_flip_main_15.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_bit_flip_main.py --bit_flip 16 > fixmatch_bit_flip_main_16.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_bit_flip_main.py --bit_flip 17 > fixmatch_bit_flip_main_17.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python -u fixmatch_bit_flip_main.py --bit_flip 18 > fixmatch_bit_flip_main_18.log 2>&1 &












# CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_bit_flip.py > fixmatch_bit_flip_new_f1.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_bit_flip.py 4 > fixmatch_bit_flip_4_new_f1.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python -u fixmatch_bit_flip.py 5 > fixmatch_bit_flip_5.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python -u fixmatch_bit_flip.py 6 > fixmatch_bit_flip_6.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python -u fixmatch_bit_flip.py 15 > fixmatch_bit_flip_15.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python -u fixmatch_bit_flip.py 16 > fixmatch_bit_flip_16.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python -u fixmatch_bit_flip.py 7 > fixmatch_bit_flip_7.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python -u fixmatch_bit_flip.py 8 > fixmatch_bit_flip_8.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_bit_flip.py 9 > fixmatch_bit_flip_9.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_bit_flip.py 10 > fixmatch_bit_flip_10.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_bit_flip.py 13 > fixmatch_bit_flip_13.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_bit_flip.py 14 > fixmatch_bit_flip_14.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python -u fixmatch_bit_flip.py 12 > fixmatch_bit_flip_12.log 2>&1 &




# nohup python -u fixmatch_bit_flip_w_al.py > fixmatch_bit_flip_w_active.log 2>&1 &

# nohup python -u fixmatch_w_active.py > fixmatch_w_active.log 2>&1 &

