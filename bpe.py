import re
import random
import collections
import pandas as pd
from tqdm import tqdm
from enum import IntEnum

class Token(IntEnum) :
    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3

def en_preprocess(sen) :  
    sen = sen.lower()
    sen = re.sub('[0-9]+,*[0-9]*', 'NUM ', sen)
    sen = re.sub('[^A-Za-z?!,.\' ]', ' ', sen)
    sen = re.sub(' {2,}' , ' ', sen)
    return sen

def kor_preprocess(sen) :
    sen = re.sub('[0-9]+,*[0-9]*', 'NUM ', sen)
    sen = re.sub('[^A-Z가-힣?!,.\' ]', ' ', sen)
    sen = re.sub(' {2,}' , ' ', sen)
    return sen

class Preprocessor :
    def __init__(self, text, preprocess, tokenize) :
        self.preprocess = preprocess
        self.tokenize = tokenize
        self.data = [tokenize(preprocess(sen)) for sen in text]

    # build data for bpe
    def get_bpe(self, merge_count) :
        counter = collections.Counter()
        for tok_list in self.data :
            counter.update(tok_list)
        counter = dict(counter)
        
        bpe_dict = {}
        subword_set = set()
        for tok, counts in counter.items() :
            ch_tuple = tuple(tok) + ('_',)
            ch_str = ' '.join(ch_tuple)
            bpe_dict[ch_str] = counts
            subword_set.update(list(ch_tuple))

        subword_list = list(subword_set)

        bpe_code = {}
        for i in tqdm(range(merge_count)) :
            pairs = self.get_stats(bpe_dict)
            if len(pairs) == 0 :
                break
            best = max(pairs, key=pairs.get)
            bpe_dict = self.merge_vocab(best, bpe_dict)
            bpe_code[best] = i
            bigram = ''.join(best)
            subword_list.append(bigram)

        subword_list = ['PAD', 'UNK', 'SOS', 'EOS'] + sorted(subword_list, key=len, reverse=True)
        return bpe_code, subword_list

    # code from paper
    def get_stats(self, vocab):
        pairs = collections.defaultdict(int) 
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i],symbols[i+1]] += freq 
        return pairs

    # code from paper
    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)') 
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word] 
        return v_out


class Tokenizer:
    def __init__(self, bpe_code, preprocess, tokenize) :
        if isinstance(bpe_code, pd.DataFrame) :
            self.set_data(bpe_code)
        else :
            self.bpe_code = bpe_code
        self.preprocess = preprocess
        self.tokenize_fn = tokenize

    def get_pairs(self, word) :
        pairs = set()
        prev_char = word[0]
        for char in word[1:] :
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    # code from paper
    def get_sub(self, orig) :
        word = tuple(orig) + ('_',)
        pairs = self.get_pairs(word)  

        if not pairs:
            return orig

        iteration = 0
        while True:
            iteration += 1
            bigram = min(pairs, key = lambda pair: self.bpe_code.get(pair, float('inf')))
            if bigram not in self.bpe_code:
                break
            first, second = bigram # first tok, second tok
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
                
        return word

    def set_data(self, bpe_data) :
        bpe_code = {}
        data_list = list(bpe_data['data'])
        count_list = list(bpe_data['count'])
        for i in range(len(bpe_data)) :
            tok_list = re.findall('\'[a-zA-Z가-힣!?.,\'_]+\'', data_list[i])
            tok_tuple = tuple([tok[1:-1] for tok in tok_list])
            count = count_list[i]
            bpe_code[tok_tuple] = count
        self.bpe_code = bpe_code

    def get_data(self) :
        return self.bpe_code

    def tokenize(self, sen) :
        sen = self.preprocess(sen)
        tok_list = self.tokenize_fn(sen)
        subword_list = []
        for tok in tok_list :
            subwords = self.get_sub(tok)
            subword_list += list(subwords)
            
        return subword_list

class Encoder :
    def __init__(self, subword_list) :
        if isinstance(subword_list, pd.DataFrame) :
            self.set_data(subword_list)
        else :
            self.sub2idx = self.build_data(subword_list)

    def build_data(self, subword_list) :
        idx = 0
        sub2idx = {}
        for tok in subword_list :
            sub2idx[tok] = idx
            idx += 1
        return sub2idx

    def get_data(self) :
        return self.sub2idx

    def set_data(self, data_df) :
        data_size = len(data_df)
        tok_list = list(data_df['token'])
        idx_list = list(data_df['index'])
        sub2idx = {}
        for i in range(data_size) :
            sub2idx[tok_list[i]] = idx_list[i]
        self.sub2idx = sub2idx

    def encode(self, tok_list) :
        idx_list = [] 
        for tok in tok_list :
            if tok in self.sub2idx :
                idx = self.sub2idx[tok]
            else :
                idx = Token.UNK
            idx_list.append(idx)
        return idx_list

