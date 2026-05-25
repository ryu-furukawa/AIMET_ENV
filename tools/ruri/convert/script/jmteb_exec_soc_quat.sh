#!/usr/bin/env bash
set -e

SECONDS=0
LOG_FILE="/root/AIMET_ENV/tools/ruri/convert/script/runtime_0522.log"


# Model Info
model=/root/AIMET_ENV/ruri-small-v2
config_file=ruri-v2-htp.json

# Dataset Info
dataset_path=/root/jmteb_260326
#dataset_name_list=(nlp_journal_title_abs)
#dataset_name_list=(nlp_journal_abs_intro nlp_journal_title_intro nlp_journal_abs_article)
#dataset_name_list=(nlp_journal_abs_intro nlp_journal_title_abs nlp_journal_title_intro nlp_journal_abs_article)
dataset_name_list=(nlp_journal_abs_intro nlp_journal_title_abs nlp_journal_title_intro nlp_journal_abs_article jagovfaqs_22k)
#dataset_name_list=(jagovfaqs_22k)
vocab_type_list=(query corpus)
#vocab_type_list=(query)
# Board Info

board_bin_dir=/data/local/tmp/htp/ruri_bin
board_lib_dir=/data/local/tmp/htp/lib


# Result Dir
sentence_txt_dir=/root/AIMET_ENV/tools/ruri/convert/script/result/sentence
vector_result_dir=/root/AIMET_ENV/tools/ruri/convert/script/result/vector_0522
pooling_result_dir=/root/AIMET_ENV/tools/ruri/convert/script/result/pooling_0522    

mkdir -p ${sentence_txt_dir}
mkdir -p ${vector_result_dir}
mkdir -p ${pooling_result_dir}

model_list=(
    ruri_w8a16_abs_mask_neg6_tf_enhanced
    ruri_w8a16_art_mask_neg6_tf_enhanced
    ruri_w8a16_mix_mask_neg6_tf_enhanced

    ruri_w8a16_abs_mask_fp32min_tf_enhanced
    ruri_w8a16_art_mask_fp32min_tf_enhanced
    ruri_w8a16_mix_mask_fp32min_tf_enhanced

    ruri_w8a16_abs_mask_neg10_tf_enhanced
    ruri_w8a16_art_mask_neg10_tf_enhanced
    ruri_w8a16_mix_mask_neg10_tf_enhanced

    ruri_w8a16_abs_mask_neg6_minmax
    ruri_w8a16_art_mask_neg6_minmax
    ruri_w8a16_mix_mask_neg6_minmax

    ruri_w8a16_abs_mask_fp32min_minmax
    ruri_w8a16_art_mask_fp32min_minmax
    ruri_w8a16_mix_mask_fp32min_minmax

    ruri_w8a16_abs_mask_neg10_minmax
    ruri_w8a16_art_mask_neg10_minmax
    ruri_w8a16_mix_mask_neg10_minmax
)

for model_name in ${model_list[@]}
do
   SECONDS=0


    vector_quat_result_dir=${vector_result_dir}/${model_name}
    pooling_quat_result_dir=${pooling_result_dir}/${model_name}
    mkdir -p ${vector_quat_result_dir}
    mkdir -p ${pooling_quat_result_dir}

    adb push /root/AIMET_ENV/tools/ruri/convert/output/script_output/${model_name}.bin /data/local/tmp/htp/ruri_bin/ruri.bin
    source /root/jmteb_data4/bin/activate
    #python preprocess_jmteb.py \
    #    --model ${model} \
    #    --dataset_path ${dataset_path} \
    #    --dataset_name_list ${dataset_name_list[@]} \
    #    --vocab_type_list ${vocab_type_list[@]} \
    #    --sentence_txt_dir ${sentence_txt_dir}

    for dataset_name in ${dataset_name_list[@]}
    do
        for vocab_type in ${vocab_type_list[@]}
        do
            sentence_file=${sentence_txt_dir}/sentence-${dataset_name}-${vocab_type}.txt
            result_dir_child=${vector_quat_result_dir}/${dataset_name}-${vocab_type}

            bash jmteb_exec_embedding_android_adoptor.sh \
                --sentence_file ${sentence_file} \
                --board_bin_dir ${board_bin_dir} \
                --board_lib_dir ${board_lib_dir} \
                --model_config_json ${config_file} \
                --result_dir ${result_dir_child}
        done
    done

    source /root/jmteb_data4/bin/activate


    python read_raw.py \
        --dataset_path ${dataset_path} \
        --dataset_name_list ${dataset_name_list[@]} \
        --vocab_type_list ${vocab_type_list[@]} \
        --raw_dir_path ${vector_quat_result_dir} \
        --result_pkl_dir ${pooling_quat_result_dir}


    echo "$(date '+%Y-%m-%d %H:%M:%S') | Total elapsed time: $((SECONDS / 60)) min $((SECONDS % 60)) sec ${model_name}" >> "${LOG_FILE}"
done