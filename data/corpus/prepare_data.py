import itertools
import copy

import numpy as np
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('../../package/bert_base_chinese')
label = {'N': 0, 'E': 1, 'C': 2, 'P': 3}


def load_data(input_file):
    print(f'load data_file: {input_file}')
    document, y_position, y_pairs, doc_len = [], [], [], []
    doc_id = []
    max_doc_len = 0

    inputFile = open(input_file, 'r')
    while True:
        line = inputFile.readline()
        if line == '':
            break
        line = line.strip().split()
        doc_id.append(line[0])
        d_len = int(line[1])
        max_doc_len = max(d_len, max_doc_len)
        pairs = eval('[' + inputFile.readline().strip() + ']')
        doc_len.append(d_len)
        y_pairs.append(pairs)
        sen_seq = []
        for i in range(d_len):
            words = inputFile.readline().strip().split(',')[-1].replace(' ', '')
            tokens = tokenizer.tokenize(words)
            token_id = tokenizer.convert_tokens_to_ids(tokens)
            sen_seq.append(token_id)
        document.append(sen_seq)

    document, y_pairs = map(np.array, [document, y_pairs])
    assert document.shape[0] == y_pairs.shape[0]
    print('load data done !')
    np.save('./all_data.npy', document)
    np.save('./pairs.npy', y_pairs)
    print('data transfer into numpy !')
    print(f'max_doc_len: {max_doc_len}')


def split_data(k_fold=1):
    document = np.load('./all_data.npy', allow_pickle=True)
    pairs = np.load('./pairs.npy', allow_pickle=True)

    number = document.shape[0]
    fold = number // 10

    d_1 = document[0: fold]
    p_1 = pairs[0: fold]

    t_d = document[fold: number]
    t_p = pairs[fold: number]

    np.save('./fold_1_test_doc.npy', d_1)
    np.save('./fold_1_test_tag.npy', p_1)
    np.save('./fold_1_train_doc.npy', t_d)
    np.save('./fold_1_train_tag.npy', t_p)

    # for i, pair in enumerate(pairs):
    #     doc_len = len(document[i])
    #     tag_metrix = np.zeros(shape=[doc_len, doc_len])
    #     doc = [sentence + [102] for sentence in document[i]]
    #     doc = [101] + list(itertools.chain(*doc))
    #     if len(doc) > 512:
    #         raise ValueError('太长了文档')
    #
    #     for p in pair:
    #         e_position = p[0] - 1
    #         c_position = p[1] - 1
    #         tag_metrix[e_position][e_position] = 1
    #         tag_metrix[c_position][c_position] = 2
    #         tag_metrix[e_position][c_position] = 3
    #         tag_metrix[c_position][e_position] = 3
    #
    #     mask = copy.deepcopy(doc)
    #     for j, idx in enumerate(mask):
    #         if idx == 102:
    #             mask[j] = 1
    #         else:
    #             mask[j] = 0



if __name__ == '__main__':
    # load_data('all_data_pair.txt')
    split_data()
