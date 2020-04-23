import os
import json
import re
import collections
import numpy as np
import pickle


def read_file(data_path, use_json=False):
    if not os.path.exists(data_path):
        raise Exception('{} don\'t exist !'.format(data_path))

    with open(data_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        while True:
            if line is None or line == '':
                break
            line = line.strip()
            if use_json:
                yield json.loads(line)
            else:
                yield line
            line = f.readline()


def print_metric(metric, style='train'):
    if style == 'train':
        print_str = '| Epoch {} - {} |'.format(metric['epoch_i'], metric['batch_i'])
    elif style == 'eval':
        print_str = '| Valid |'
    else:
        print_str = ''
    for k, v in metric.items():
        if k in ['epoch_i', 'batch_i']:
            continue
        print_str += ' {}: {:.4f} |'.format(k, v)
    print(print_str)
    # return print_str


def get_vocab(data_path, vocab_size=None, cut_freq=1):
    tokens = []
    total = 0
    for _ in data_path:
        for o in read_file(_, use_json=True):
            text = o['text'].strip()
            text = re.sub(r'\s', '', text)
            for t in text:
                total += 1
                if re.search(r'[\u0800-\u4e00\uAC00-\uD7A3]', t):
                    continue
                tokens.append(t)
    counter = collections.Counter(tokens)
    tokens = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    # for _, freq in tokens:
    #     total += freq

    tags = ['<PAD>', '<S>', '<E>', '<UNK>']
    # with open('./dict.txt', 'w') as f:
    for _ in tags:
        yield _, 0, 0
        # f.write('{}\t{}\t{:.4f}\n'.format(_, 0, 0))
    count = 0
    for word, freq in tokens:
        if freq < cut_freq:
            break
        count += freq
        yield word, freq, count / total
        # f.write('{}\t{}\t{:.4f}\n'.format(word, freq, count / total))


def get_schema(data_path, out_dir):
    print('load schema ...')
    out_path = os.path.join(out_dir, 'schema.pkl')
    if os.path.exists(out_path):
        with open(out_path, 'rb') as f:
            theme, role, form, relation = pickle.load(f)
    else:
        theme = dict()
        role = dict()
        form = dict()
        role['s'] = len(role)
        for o in read_file(data_path, use_json=True):
            p = o['predicate']
            ob = o['object_type']
            sub = o['subject_type']
            if p not in theme:
                theme[p] = len(theme)
            else:
                continue
            if sub not in form:
                form[sub] = len(form)
            for k, v in ob.items():
                _tmp = 'o-{}'.format(k)
                if _tmp not in role:
                    role[_tmp] = len(role)
                if v not in form:
                    form[v] = len(form)

        relation = np.zeros(shape=(len(theme), len(role), len(form)), dtype=np.int)
        for o in read_file(data_path, use_json=True):
            p = o['predicate']
            ob = o['object_type']
            sub = o['subject_type']
            _tmp = relation[theme[p]]
            _tmp[role['s']][form[sub]] = 1
            for k, v in ob.items():
                _r = 'o-{}'.format(k)
                _tmp[role[_r]][form[v]] = 1

        with open(out_path, 'wb') as f:
            pickle.dump([theme, role, form, relation], f)

    print('| theme: {} | role: {} | form: {} |'.format(len(theme), len(role), len(form)))
    return theme, role, form, relation


class SPOClassifyData:
    def __init__(self, data_dir, out_dir):
        data_train_path = os.path.join(data_dir, 'train_data/train_data.json')
        data_dev_path = os.path.join(data_dir, 'dev_data/dev_data.json')
        schema_path = os.path.join(data_dir, 'schema.json')
        # get vocab
        vocab = dict()
        vocab_path = os.path.join(out_dir, 'vocab.txt')
        if os.path.exists(vocab_path):
            for line in read_file(vocab_path):
                token, _, _ = line.split('\t')
                vocab[token] = len(vocab)
        else:
            with open(vocab_path, 'w', encoding='utf-8') as f:
                for token, freq, rate in get_vocab([data_train_path, data_dev_path], cut_freq=50):
                    vocab[token] = len(vocab)
                    f.write('{}\t{}\t{}\n'.format(token, freq, rate))
        self.vocab = vocab
        # get schema
        self.schema = get_schema(schema_path, out_dir)
        # print(self.schema[0])
        train_path = os.path.join(out_dir, 'train.dat')
        dev_path = os.path.join(out_dir, 'dev.dat')
        self.train_data = self.get_train_data(data_train_path, train_path, vocab)
        self.dev_data = self.get_train_data(data_dev_path, dev_path, vocab)

    def get_train_data(self, data_path, out_path, vocab):
        temp_data = []
        if os.path.exists(out_path):
            for o in read_file(out_path, use_json=True):
                temp_data.append(o)
        else:
            out_f = open(out_path, 'w', encoding='utf-8')
            for o in read_file(data_path, use_json=True):
                x = list()
                x.append(vocab['<S>'])
                for _ in o['text']:
                    if _ in vocab:
                        x.append(vocab[_])
                    else:
                        x.append(vocab['<UNK>'])
                x.append(vocab['<E>'])
                y = []
                for _ in o['spo_list']:
                    y.append(self.schema[0][_['predicate']])
                temp_data.append({'x': x, 'y': y})
                out_f.write(json.dumps({'x': x, 'y': y}, ensure_ascii=False) + '\n')
            out_f.close()
        return temp_data

    def get_batch_generator(self, batch_size, seq_length):
        out_size = len(self.schema[0])

        def batch_generator(in_data):
            size = len(in_data)
            cursor = batch_size
            is_over = False
            text_ids = [_ for _ in range(batch_size)]
            buffers = [0 for _ in range(batch_size)]
            while True:
                batch_x = np.zeros(shape=(batch_size, seq_length), dtype=np.int)
                batch_y = np.zeros(shape=(batch_size, out_size), dtype=np.int)
                mask = np.zeros(shape=(batch_size, out_size), dtype=np.int)
                for i in range(batch_size):
                    d = in_data[text_ids[i]]
                    text = d['x']
                    if len(text[buffers[i]:]) <= seq_length:
                        batch_x[i, :len(text[buffers[i]:])] = text[buffers[i]:]
                        batch_y[i, d['y']] = 1
                        # mask[i, len(text[buffers[i]:]) - 1] = 1
                        mask[i] = np.ones(shape=(out_size,), dtype=np.int)
                        if cursor >= size:
                            is_over = True
                            break
                        buffers[i] = 0
                        text_ids[i] = cursor
                        cursor += 1
                    else:
                        batch_x[i] = text[buffers[i]: buffers[i] + seq_length]
                        buffers[i] += seq_length
                if is_over:
                    break
                yield batch_x, batch_y, mask
                # yield batch_x, batch_y

        return batch_generator(self.train_data), batch_generator(self.dev_data)


if __name__ == '__main__':
    # get_input_vocab('/Users/boss/Documents/data/百度事件抽取/train_data/train_data.json')
    # get_output_vocab('/Users/boss/Documents/data/百度事件抽取/schema.json', './')
    data_loader = SPOClassifyData('../data', './data')
    # train_generator, dev_generator = data_loader.get_batch_generator(32, 64)
    # i = 0
    # for x, y, mask in train_generator:
    #     if i % 100 == 0:
    #         print('load {} batch'.format(i))
    #     i += 1