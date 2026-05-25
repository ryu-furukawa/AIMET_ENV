#!/usr/bin/env bash
set -e

#source /root/teb3/bin/activate

cd /root/JMTEB

source /root/jmteb_data4/bin/activate
export DATASET_PATH="/root/jmteb_260326"
#export DATASET_PATH="/root/jmteb_1104"
#export MODEL=/root/AIMET_ruri_v2/ruri-small-v2-onnx
#export MODEL=/root/adoptor/ruri-small-v2_supervised_adaptor
export MODEL=/root/AIMET_ENV/tools/ruri/ruri-small-v2
#export EVAL_DATASET="['mrtydi']"
#export EVAL_DATASET="['nlp_journal_abs_intro']"
#export EVAL_DATASET="['jagovfaqs_22k']"
#export EVAL_DATASET="['nlp_journal_title_abs','nlp_journal_title_intro','nlp_journal_abs_intro','nlp_journal_abs_article']"

#export EVAL_DATASET="['nlp_journal_title_abs','nlp_journal_title_intro','nlp_journal_abs_intro','jagovfaqs_22k']"
export EVAL_DATASET="['nlp_journal_title_abs','nlp_journal_title_intro','nlp_journal_abs_intro','nlp_journal_abs_article','jagovfaqs_22k']"

#dim_list=(768 512 384 256 192 128 64 32)
dim_list=(768)
for dim in "${dim_list[@]}"; do
    poetry run python -m jmteb  --embedder SentenceBertEmbedder   \
                                --embedder.model_name_or_path $MODEL   \
                                --save_dir "/root/AIMET_ENV/tools/ruri/convert/script/jmteb_result/jmteb_pc/dim_$dim"   \
                                --eval_include "$EVAL_DATASET" \
                                --truncate_dim $dim \
                                --log_predictions true \
                                --dataset_path $DATASET_PATH
                                
done
