import json
import stanza
import joblib
import string
import itertools
import numpy as np
from tqdm import tqdm
from collections import Counter
from nltk.corpus import stopwords
from argparse import ArgumentParser
from stanza.models.constituency.parse_tree import Tree

stanza_path = "/data/Installer/stanza-en"
embedding_dim = 300


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='Dataset name.', type=str, default='mr')  # mr, ohsumed, 20ng, R8, R52
    parser.add_argument('--cut_len', help='Truncate the sentence to a specified length.', type=int, default=512)
    args = parser.parse_args()
    return args


def mask_words(token_word_list, stop_words, word2count=None, least_freq=1):
    """
    Mask words which is stop word or punctuation || or rare word -- abandoned (not as frequent as least_freq)
    :param token_word_list:
    :param stop_words:
    :param word2count: not use
    :param least_freq: not use
    :return:
    """
    token_mask_list = []
    for word in token_word_list:
        # word_freq = word2count[word]
        if word not in stop_words:  # and word_freq >= least_freq
            token_mask_list.append(1)
        else:
            token_mask_list.append(0)
    return token_mask_list


def get_depen_edge(sent, sent_shift, prev_root_id, anti_rel):
    """
    Get dependency edges
    :param sent: sentence parsed by stanza
    :param sent_shift: id shift of words between sentences
    :param prev_root_id: root word id of the previous sentence
    :param anti_rel: dependency relation type that will be ignored
    :return: depen_edges: dependency edges within this sentence
    """
    depen_edges = []
    this_root_id = 0
    for dep_edge in sent.dependencies:
        dep_rel = dep_edge[1]
        if dep_rel == 'root':  # add cross-sentence edges between each root node
            this_root_id = dep_edge[2].id - 1 + sent_shift
            # this_root = dep_edge[2].text
            if prev_root_id != -1:
                # print(r"{} ({}) --> [{}] --> {} ({})".format(
                #     prev_root, prev_root_id, dep_rel, this_root, this_root_id))
                depen_edges.append([prev_root_id, dep_rel, this_root_id])
            # prev_root_id = this_root_id
            # prev_root = this_root
        elif dep_rel not in anti_rel:
            head_id = dep_edge[0].id - 1 + sent_shift
            tail_id = dep_edge[2].id - 1 + sent_shift
            # head = dep_edge[0].text
            # tail = dep_edge[2].text
            # print(r"{} ({}) --> [{}] --> {} ({})".format(
            #     head, head_id, dep_rel, tail, tail_id))
            depen_edges.append([head_id, dep_rel, tail_id])
    return depen_edges, this_root_id


def traverse_tree(tree: Tree, const_words: list):
    """
    Traverse the constituency tree
    :param tree: constituency tree
    :param const_words: words within each constituency
    :return:
    """
    if tree.depth() <= 3:
        const_words.append(tree.leaf_labels())
        return
    if not tree.is_leaf():
        for child in tree.children:
            traverse_tree(child, const_words)


def merge_isolate(const_words):
    """
    Merge isolated word between each constituency, except for punctuations
    :param const_words: words within each constituency
    :return:
    """
    new_const_words = []
    temp_const = []
    for words in const_words:
        if len(words) == 1:
            if words[0] not in string.punctuation:
                temp_const.append(words[0])
            else:  # isolated punctuation
                if len(temp_const) > 0:
                    new_const_words.append(temp_const)
                    temp_const = []
                new_const_words.append(words)
        else:
            if len(temp_const) > 0:
                new_const_words.append(temp_const)
                temp_const = []
            new_const_words.append(words)
    return new_const_words


def get_const_edge_old(sent, sent_shift):
    """
    Get constituency edges
    :param sent: sentence parsed by stanza
    :param sent_shift: id shift of words between sentences
    :return: const_edges: constituency edges within this sentence
    """
    const = sent.constituency
    const_words = []
    traverse_tree(const, const_words)
    const_words = merge_isolate(const_words)
    # print(const_words)
    windows_size = 3
    const_shift = 0
    const_edges = []
    for words in const_words:
        const_size = len(words)
        # skip the constituency which only have one word
        if const_size > 1:
            for i in range(const_size - 1):
                for j in range(i + 1, i + windows_size):  # get edges within the windows
                    if j == const_size:
                        break
                    const_edges.append([i + const_shift + sent_shift, j + const_shift + sent_shift])
        const_shift += const_size
    return const_edges


def get_const_edge(sent, sent_shift):
    """
    Get constituency edges
    :param sent: sentence parsed by stanza
    :param sent_shift: id shift of words between sentences
    :return: const_edges: constituency edges within this sentence
    """
    const = sent.constituency
    const_words = []
    traverse_tree(const, const_words)
    const_words = merge_isolate(const_words)
    # print(const_words)
    const_shift = 0
    const_edges = []
    tail_word_id = -1
    for words in const_words:
        head_word_id = const_shift
        if tail_word_id > 0:
            const_edges.append([tail_word_id + sent_shift, head_word_id + sent_shift])
        const_size = len(words)
        tail_word_id = head_word_id + const_size - 1
        # skip the constituency which only have one word
        if const_size > 1:
            id_pairs = list(itertools.combinations(range(const_size), 2))
            for i, j in id_pairs:
                const_edges.append([i + const_shift + sent_shift, j + const_shift + sent_shift])
        const_shift += const_size
    # print(const_edges)
    return const_edges


def get_truncate(parse_dict, cut_len):
    doc_id = parse_dict['id']
    token_word_list = parse_dict['token_list']
    unpos_type_list = parse_dict['unpos_type']
    depen_edge_list = parse_dict['depen_edge']
    const_edge_list = parse_dict['const_edge']

    new_token_word_list = token_word_list[:cut_len]
    new_unpos_type_list = unpos_type_list[:cut_len]
    new_depen_edge_list = [elem for elem in depen_edge_list if elem[0] < cut_len and elem[2] < cut_len]
    new_const_edge_list = [elem for elem in const_edge_list if elem[0] < cut_len and elem[1] < cut_len]
    new_parse_dict = {
        'id': doc_id,
        'token_list': new_token_word_list,
        'unpos_type': new_unpos_type_list,
        'depen_edge': new_depen_edge_list,
        'const_edge': new_const_edge_list
    }
    return new_parse_dict


def apply_mask_trunc(parse_result, word2count, stop_words, least_freq, cut_len=512):
    new_parse_result = list()
    for doc_id in range(len(parse_result)):
        token_word_list = parse_result[doc_id]['token_list']
        unpos_type_list = parse_result[doc_id]['unpos_type']
        depen_edge_list = parse_result[doc_id]['depen_edge']
        const_edge_list = parse_result[doc_id]['const_edge']

        new_token_word_list = []
        new_unpos_type_list = []
        # Traverse the word list and construct a new index mapping
        new_index = {}
        new_id = 0
        for i in range(len(token_word_list)):
            this_pos = unpos_type_list[i]
            this_word = token_word_list[i]
            word_freq = word2count[this_word]
            if this_word not in stop_words and word_freq >= least_freq:
                new_index[i] = new_id
                new_id += 1
                new_token_word_list.append(this_word)
                new_unpos_type_list.append(this_pos)
        new_depen_edge_list = [(new_index[src_id], rel_type, new_index[trg_id])
                               for src_id, rel_type, trg_id in depen_edge_list
                               if src_id in new_index and trg_id in new_index]
        new_const_edge_list = [(new_index[src_id], new_index[trg_id])
                               for src_id, trg_id in const_edge_list
                               if src_id in new_index and trg_id in new_index]
        parse_dict = {
            'id': doc_id,
            'token_list': new_token_word_list,
            'unpos_type': new_unpos_type_list,
            'depen_edge': new_depen_edge_list,
            'const_edge': new_const_edge_list
        }
        new_parse_result.append(get_truncate(parse_dict, cut_len))
    return new_parse_result


def parse_text(text_list, nlp, stop_words):
    print("Parse text with Stanza:")
    # todo: complete anti relations (refer to TextFCG)
    # anti_rel = ['aux', 'aux:pass', 'case', 'cc', 'cop', 'det', 'fixed', 'mark', 'nummod', 'punct']
    anti_rel = ['punct']
    parse_result = []
    for doc_id, text in enumerate(tqdm(text_list, ascii=True)):
        doc = nlp(text)
        token_word_list = []
        unpos_type_list = []
        depen_edge_list = []
        const_edge_list = []
        sent_shift = 0
        prev_root_id = -1
        for sent_id, sent in enumerate(doc.sentences):
            tokened_words = [word.text for word in sent.words]
            token_word_list.extend(tokened_words)
            universal_pos = [word.upos for word in sent.words]
            unpos_type_list.extend(universal_pos)
            depen_edges, this_root_id = get_depen_edge(sent, sent_shift, prev_root_id, anti_rel)
            depen_edge_list.extend(depen_edges)
            prev_root_id = this_root_id
            const_edges = get_const_edge(sent, sent_shift)
            const_edge_list.extend(const_edges)
            sent_shift += len(tokened_words)
        token_mask_list = mask_words(token_word_list, stop_words)
        parse_dict = {
            'id': doc_id,
            'token_list': token_word_list,
            'token_mask': token_mask_list,
            'unpos_type': unpos_type_list,
            'depen_edge': depen_edge_list,
            'const_edge': const_edge_list
        }
        # reserve position for placeholders [CLS] & [SEP]
        # parse_dict = get_truncate(parse_dict, cut_len - 2)
        parse_result.append(parse_dict)
    return parse_result


def build_word2index(parse_result):
    # remake word2index
    print("Get word2index.")
    all_words_list = []
    for doc_id in range(len(parse_result)):
        word_node_list = parse_result[doc_id]['token_list']
        for w in word_node_list:
            all_words_list.append(w)
    word2count = Counter(all_words_list)
    word_count = [[w, c] for w, c in word2count.items()]
    word2index = {w: i for i, (w, c) in enumerate(word_count)}
    return word2index


def build_word2vec(word2index):
    def get_oov():
        oov = np.random.normal(-0.1, 0.1, embedding_dim)
        return oov

    # build word2vec
    print("Build word2vec.")
    all_vectors = np.load(f"source/glove.6B.{embedding_dim}d.npy")
    all_words = joblib.load(f"source/glove.6B.words.pkl")
    all_word2index = {w: i for i, w in enumerate(all_words)}
    word_set = [w for w, i in word2index.items()]
    word2vec = [all_vectors[all_word2index[w]] if w in all_word2index else get_oov() for w in word_set]
    # add an all-zero vector at the end of the vocab as the vector of [PAD]
    word2vec.append(np.zeros(embedding_dim))
    return word2vec


def build_pos2vec(parse_result):
    pos_bag = list()
    for doc_id in range(len(parse_result)):
        unpos_type_list = parse_result[doc_id]['unpos_type']
        pos_bag.extend(unpos_type_list)
    pos2count = Counter(pos_bag)
    pos_count = [[p, c] for p, c in sorted(pos2count.items())]
    pos2index = {p: i for i, (p, c) in enumerate(pos_count)}
    return pos2index


def build_word2count(parse_result):
    word_bag = list()
    for doc_id in range(len(parse_result)):
        token_word_list = parse_result[doc_id]['token_list']
        word_bag.extend(token_word_list)
    word2count = Counter(word_bag)
    return word2count


def main():
    args = get_args()
    dataset = args.dataset

    nlp = stanza.Pipeline(lang='en', processors='tokenize, mwt, pos, lemma, depparse, constituency',
                          tokenize_no_ssplit=False, use_gpu=True, model_dir=stanza_path, download_method=None)

    text_list = []
    with open('data/temp/{}'.format(dataset) + '.texts.remove.txt', 'r') as file:
        for line in file.readlines():
            text_list.append(line.strip())

    print('Start parsing text of dataset:', dataset)
    stop_words = set(stopwords.words('english')).union(set(string.punctuation))
    if dataset == 'mr':
        stop_words = set(string.punctuation)

    parse_result = parse_text(text_list, nlp, stop_words)

    word2index = build_word2index(parse_result)
    joblib.dump(word2index, f"data/temp/{dataset}.word2index.pkl")
    pos2index = build_pos2vec(parse_result)
    joblib.dump(pos2index, f"data/temp/{dataset}.pos2index.pkl")
    word2vec = build_word2vec(word2index)
    np.save(f"data/temp/{dataset}.word2vec.npy", word2vec)

    with open(f"data/temp/{dataset}.parse.json", 'w', encoding='utf-8') as f:
        json.dump(parse_result, f, ensure_ascii=False)  # new_parse_result


if __name__ == '__main__':
    main()
