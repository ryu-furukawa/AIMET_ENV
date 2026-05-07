#!/usr/bin/env bash
set -e

source /root/jmteb_data4/bin/activate

cd /root/JMTEB
export pkl_dir="/root/AIMET_ENV/tools/ruri/convert/script/result/pooling_quat"
export MODEL=/root/AIMET_ruri/ruri-small-v2
#export MODEL=/root/t2e/ruri-small-v2
#export EVAL_DATASET="['mrtydi']"
#export EVAL_DATASET="['jagovfaqs_22k']"
#export EVAL_DATASET="['nlp_journal_title_abs','nlp_journal_title_intro','nlp_journal_abs_intro','nlp_journal_abs_article']"
#export EVAL_DATASET="['nlp_journal_abs_intro']"
export EVAL_DATASET="['nlp_journal_title_abs','nlp_journal_title_intro','nlp_journal_abs_intro','nlp_journal_abs_article','jagovfaqs_22k']"
#dim_list=(768 512 384 256 192 128 64 32)
model_list=(
    ruri_w8a16_PT_htp_v73 
    ruri_w8a8_PT_htp_v73 
    ruri_w8a16_PT 
    ruri_w8a8_PT 
    ruri_w8a16_minmax_htp_v73 
    ruri_w8a8_minmax_htp_v73 
    ruri_w8a16_minmax 
    ruri_w8a8_minmax
)

for model_name in ${model_list[@]}; do
    poetry run python -m jmteb  --embedder SentenceBertEmbedder   \
                                --embedder.model_name_or_path $MODEL   \
                                --save_dir "/root/AIMET_ENV/tools/ruri/convert/script/jmteb_result/jmteb_soc_quat/$model_name"   \
                                --eval_include "$EVAL_DATASET" \
                                --pkl_path "$pkl_dir/$model_name" \
                                --log_predictions true \
                                --dataset_path "/root/jmteb_260326"
                                
done
