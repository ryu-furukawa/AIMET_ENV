import os
import argparse
import torch
import unicodedata
from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk

def preprocess_text(inputs):

    # Copy from distilbert_japanese_tokenizer.py > SentencepieceTokenizer class
    remove_space = True
    keep_accents = True
    do_lower_case = False

    if remove_space:
        outputs = " ".join(inputs)
    else:
        outputs = inputs
    outputs = outputs.replace("``", '"').replace("''", '"')

    if not keep_accents:
        outputs = unicodedata.normalize("NFKD", outputs)
        outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
    if do_lower_case:
        outputs = outputs.lower()

    outputs = outputs.replace(" \n", "").replace(" \r", "").replace(" \r\n", "")

    return outputs

def mecab_process(sentences):

    # unidic=1.1.0, unidic-lite=1.0.8で動作確認済み。
    import fugashi
    import unidic_lite

    if isinstance(sentences, list):
        # Convert list to string
        sentences = " ".join(sentences)

    mecab_option = ""
    do_lower_case = True
    dic_dir = unidic_lite.DICDIR
    mecabrc = os.path.join(dic_dir, "mecabrc")
    mecab_option = f'-d "{dic_dir}" -r "{mecabrc}" ' + mecab_option

    mecab = fugashi.GenericTagger(mecab_option)

    text = unicodedata.normalize("NFKC", sentences)
    
    tokens = []

    for word in mecab(text):
        token = word.surface

        if do_lower_case:
            token = token.lower()

        tokens.append(token)

    add_space_token = preprocess_text(tokens)

    return add_space_token


def preprocessing(model, tokenizer, sentences):

    mecab_sentences = mecab_process(sentences)

    encoded = tokenizer.batch_encode_plus([mecab_sentences], padding=True, truncation=True)
    input_ids = encoded["input_ids"]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    text_token = ''.join(tokens[1:-1]).replace('▁', ' ').strip()

    return text_token


def main(model, dataset_path, dataset_name_list, vocab_type_list, sentence_txt_dir):


    # Load tokenizer from setting path & model from HuggingFace Hub
    model_tokenizer    = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    
    for vocab_type in (vocab_type_list):

        if vocab_type == "query":
            prefix = "クエリ: "
            split_type = "test"
            dataset_type = "query"
        elif vocab_type == "corpus":
            prefix = "文章: "
            split_type = "corpus"
            dataset_type = "text"


        for dataset_name in (dataset_name_list):
            dataset = load_from_disk(os.path.join(dataset_path, f"{dataset_name}-{vocab_type}"))[split_type]
            #dataset = load_from_disk(os.path.join(dataset_path))
            sentence_txt_path = os.path.join(sentence_txt_dir, f"sentence-{dataset_name}-{vocab_type}.txt")
            if os.path.exists(sentence_txt_path):
                os.remove(sentence_txt_path)

            for num in range(len(dataset)):

                print("=====================================================")
                print(f" {dataset_name}, ID {num}")
                print("-----------------------------------------------------")
                
                sentences = prefix + dataset[dataset_type][num]
                text_token = preprocessing(model, model_tokenizer, sentences)

                with open(sentence_txt_path, "a") as fileobj:
                    fileobj.write(text_token + "\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str)
    parser.add_argument('--dataset_path', default=None, type=str)
    parser.add_argument('--dataset_name_list', default=None, type=str, nargs='+')
    parser.add_argument('--vocab_type_list', default=None, type=str, nargs='+')
    parser.add_argument('--sentence_txt_dir', default=None, type=str)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
