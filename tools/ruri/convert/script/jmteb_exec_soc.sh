#!/usr/bin/env bash
set -e

SECONDS=0
LOG_FILE="/root/AIMET_ENV/tools/ruri/convert/script/runtime_htp.log"


# Model Info
model=/root/ruri-small-v2
config_file=ruri-v2-htp.json

# Dataset Info
dataset_path=/root/jmteb_260326
#dataset_path=/root/jmteb_1104
#dataset_path=/root/jaqket_250708
dataset_name_list=(nlp_journal_abs_intro)
#dataset_name_list=(nlp_journal_abs_intro nlp_journal_title_abs nlp_journal_title_intro nlp_journal_abs_article)
#dataset_name_list=(nlp_journal_abs_intro nlp_journal_title_abs nlp_journal_title_intro nlp_journal_abs_article jagovfaqs_22k)
#dataset_name_list=(jagovfaqs_22k)
#dataset_name_list=(jaqket)
vocab_type_list=(query corpus)
#vocab_type_list=(query)
# Board Info

board_bin_dir=/data/local/tmp/htp/ruri_bin
board_lib_dir=/data/local/tmp/htp/lib


# Result Dir
sentence_txt_dir=/root/AIMET_ENV/tools/ruri/convert/script/result/sentence
vector_result_dir=/root/AIMET_ENV/tools/ruri/convert/script/result/vector_quat
pooling_result_dir=/root/AIMET_ENV/tools/ruri/convert/script/result/pooling_quat

mkdir -p ${sentence_txt_dir}
mkdir -p ${vector_result_dir}
mkdir -p ${pooling_result_dir}

#source /root/teb3/bin/activate
#source /root/ruri/bin/activate
#source /root/JMTEB2/jmteb2.0/bin/activate
source /root/jmteb_data4/bin/activate
python preprocess_jmteb.py \
    --model ${model} \
    --dataset_path ${dataset_path} \
    --dataset_name_list ${dataset_name_list[@]} \
    --vocab_type_list ${vocab_type_list[@]} \
    --sentence_txt_dir ${sentence_txt_dir}

for dataset_name in ${dataset_name_list[@]}
do
    for vocab_type in ${vocab_type_list[@]}
    do
        sentence_file=${sentence_txt_dir}/sentence-${dataset_name}-${vocab_type}.txt
        result_dir_child=${vector_result_dir}/${dataset_name}-${vocab_type}

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
    --raw_dir_path ${vector_result_dir} \
    --result_pkl_dir ${pooling_result_dir}


echo "$(date '+%Y-%m-%d %H:%M:%S') | Total elapsed time: $((SECONDS / 60)) min $((SECONDS % 60)) sec" >> "${LOG_FILE}"