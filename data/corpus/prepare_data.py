from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('../../package/bert_base')


def load_data(input_file):
    print(f'load data_file: {input_file}')
    document, y_position, y_pairs, doc_len = [], [], [], []
    doc_id = []

    inputFile = open(input_file, 'r')
    while True:
        line = inputFile.readline()
        if line == '':
            break
        line = line.strip().split()
        doc_id.append(line[0])
        d_len = int(line[1])
        pairs = eval('[' + inputFile.readline().strip() + ']')
        doc_len.append(d_len)
        y_pairs.append(pairs)
        sen_seq = []
        for i in range(d_len):
            words = inputFile.readline().strip().split(',')[-1].replace(' ', '')
            sen_seq.append(words)
        document.append(sen_seq)


if __name__ == '__main__':
    load_data('all_data_pair.txt')
