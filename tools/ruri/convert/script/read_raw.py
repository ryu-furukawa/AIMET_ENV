import os
import torch
import numpy as np
#import MeCab
import difflib
import re
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk

from sentencepiece import SentencePieceProcessor
import unicodedata
import pickle
import argparse

def read_raw_file(file_path, model_dim):
    
    # Checking Path
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    # Read the raw file with specified dimensions
    with open(file_path, 'rb') as file:
        
        buffer = np.frombuffer(file.read(), dtype=np.float32)

        # dimensionsを計算する
        dimensions = int(len(buffer) / model_dim)

        # reshaped_bufferを作成する
        reshaped_buffer = buffer.reshape(dimensions, model_dim)


    # Convert to tensor and normalize
    #sentence_embeddings = torch.mean(torch.tensor(reshaped_buffer), dim=0)
    #平均済みの場合
    sentence_embeddings = torch.tensor(reshaped_buffer).reshape(-1)

    return sentence_embeddings


def main(dataset_path, dataset_name_list, vocab_type_list, raw_dir_path, result_pkl_dir, model_dim):

    for vocab_type in vocab_type_list:

        if vocab_type == "query":
            prefix = "クエリ: "
            split_type = "test"
            dataset_type = "query"
        elif vocab_type == "corpus":
            prefix = "文章: "
            split_type = "corpus"
            dataset_type = "text"

        for dataset_name in dataset_name_list:

            pkl_result_list_tensor = []

            dataset = load_from_disk(os.path.join(dataset_path, f"{dataset_name}-{vocab_type}"))[split_type]

            for id in range(len(dataset)):
                print("=====================================================")
                print(f" {dataset_name}, ID {id}")
                print("-----------------------------------------------------")

                raw_file_path = os.path.join(os.path.join(raw_dir_path, f"{dataset_name}-{vocab_type}"), f"output_{str(id)}.raw")
                
                read_raw_embedding = read_raw_file(raw_file_path, model_dim)

                pkl_result_list_tensor.append(read_raw_embedding)
    
            save_pkl_path = os.path.join(result_pkl_dir, f"{dataset_name}_{vocab_type}.pkl")
    
            with open(save_pkl_path, 'wb') as f:
                pkl_result_list_tensor = torch.stack(pkl_result_list_tensor)
                pickle.dump(pkl_result_list_tensor, f)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default=None, type=str)
    parser.add_argument('--dataset_name_list', default=None, type=str, nargs='+')
    parser.add_argument('--vocab_type_list', default=None, type=str, nargs='+')
    parser.add_argument('--raw_dir_path', default=None, type=str)
    parser.add_argument('--result_pkl_dir', default="./", type=str)
    parser.add_argument('--model_dim', default=768, type=int)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
