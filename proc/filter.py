# import nltk
# nltk.download("stopwords")
# from nltk.corpus import stopwords
from collections import Counter
import re
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='Dataset name.', type=str, default='R52')  # mr, ohsumed, 20ng, R8, R52
    args = parser.parse_args()
    return args


# func load texts & labels
def load_dataset(dataset_name):
    with open(f"data/raw/{dataset_name}.texts.txt", "r", encoding="latin1") as file:
        text_list = file.read().strip().split("\n")
    with open(f"data/raw/{dataset_name}.labels.txt", "r") as file:
        label_list = file.read().strip().split("\n")
    return text_list, label_list


def filter_text(text: str, dataset_name):
    if dataset_name == '20ng':
        # modify email head
        regexes_to_modify = [r' From:', r' Re:', r' Subject:', r' FAQ:', r' Summary:', r' Keywords:', r' Expires:',
                             r' Supersedes:', r' Archive-name:', r' Last-modified:', r' Version:', r' Mime-Version:',
                             r' Distribution:', r' Organization:', r' Lines:', r' Reply-To:', r' writes: >']
        for r in regexes_to_modify:
            text = re.sub(r, '.'+r, text)
        text = re.sub(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", "", text)  # remove email address
        text = re.sub(r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", "", text)  # rm url
        text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "", text)  # remove ip address
        text = re.sub(r"\d{1,2}:\d{2}(:\d{2})?", "", text)  # remove time stamp
        text = re.sub(r"\d+(-\d+)", "", text)  # remove tel
        text = re.sub(r"\(\d+\)", "", text)  # remove tel
        text = re.sub(r"\d+(\.\d+)", "", text)  # remove decimal
    elif dataset_name == 'ohsumed':
        text = re.sub(r"\(.*?\)", "", text)  # remove the contents in brackets
        text = re.sub(r"\d+(\.\d+)", "", text)  # remove decimal
    elif dataset_name == 'AGNews':
        text = re.sub(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", "", text)  # remove email address
        text = re.sub(r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", "", text)  # rm url

    text = re.sub(r"[^A-Za-z0-9(),.!?\':]", " ", text)
    text = text.replace("'m ", " am ")
    text = text.replace(r"'s", " is ")
    text = text.replace(r"'re", " are ")
    text = text.replace(r"'ll", " will ")
    text = text.replace(r"'d", " would ")
    text = text.replace(r"'ve", " have ")
    text = text.replace(" can't ", " can not ")
    text = text.replace(" ain't ", " are not ")
    text = text.replace(r"n't", " not ")
    text = text.replace("(", " ( ")
    text = text.replace(")", " ) ")
    text = text.replace(",", " , ")
    text = text.replace(".", " . ")
    text = text.replace("!", " ! ")
    text = text.replace("?", " ? ")
    text = text.replace(":", " : ")

    word_list = text.strip().lower().split()
    text = " ".join(word_list)
    text = re.sub(r"(\W\s)\1+", "\\1", text)  # remove repeated punctuations
    text = re.sub(r'\s+([,.!?\':])', r'\1', text)  # remove spaces before punctuations
    text = re.sub(r'\(\s*\)', '', text)  # remove empty brackets
    text = re.sub(r'\(\s+|\s+\)', lambda x: '(' if x.group() == '( ' else ')',
                  text)  # remove spaces before & after brackets
    return text


def main():
    args = get_args()
    dataset = args.dataset

    cut_len = 256  # truncate text, 256 by default
    # To not destroy the sentence structure, stop words and rare words are reserved
    least_freq = 0
    stop_words = set()

    print('Load dataset:', dataset)
    texts, labels = load_dataset(dataset)

    # handle labels
    label2index = {l: i for i, l in enumerate(sorted(set(labels)))}
    targets = [label2index[lb] for lb in labels]
    np.save(f"data/temp/{dataset}.targets.npy", targets)

    # handle texts
    print('Filtering text...')
    texts_clean = []
    for t in tqdm(texts, ascii=True):
        texts_clean.append(filter_text(t, dataset))  # segmentation, lowercase, standardized

    word2count = Counter([w for t in texts_clean for w in t.split()])
    word_count = [[w, c] for w, c in word2count.items()
                  if c >= least_freq and w not in stop_words]  # remove stop words & rare words
    word2index = {w: i for i, (w, c) in enumerate(word_count)}
    words_list = [[w for w in t.split() if w in word2index] for t in texts_clean]

    # no truncate
    words_list = [ws[:cut_len] for ws in words_list]
    texts_remove = [" ".join(ws) for ws in words_list]
    len_list = [len(ws) for ws in words_list]
    print('Average len:', np.mean(len_list))
    print('Vocab size:', len(word2count))

    # save
    with open(f"data/temp/{dataset}.texts.remove.txt", "w") as f:
        f.write("\n".join(texts_remove))


if __name__ == '__main__':
    main()
