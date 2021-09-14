import re
import random
import collections
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
    def __init__(self, data, preprocess, tokenize, th=3) :
        self.preprocess = preprocess
        self.tokenize = tokenize
        self.th = th
        self.tok_data = [tokenize(preprocess(sen)) for sen in data]
        
        self.build_data()

    def __len__(self) :
        return len(self.token_data)
        
    def build_data(self) :
        vocab_set = collections.Counter()
        for sen in self.tok_data :
            vocab_set.update(sen)
        
        vocab_set = dict(vocab_set)
        valid_tok = []
        for tok, count in vocab_set.items() : 
            if count >= self.th :
                valid_tok.append(tok)
                
        random.shuffle(valid_tok)
        tok_list = ['PAD', 'UNK', 'SOS', 'EOS'] + valid_tok
        
        self.word2idx = dict(zip(tok_list, range(len(tok_list))))
        self.idx2word = {word: idx for idx, word in self.word2idx.items()}
        
    def get_data(self) :
        return self.word2idx
    
    def set_data(self, data_df) :
        data_token = list(data_df['TOKEN'])
        data_index = list(data_df['INDEX'])
        
        data_size = len(data_df)
        word2idx = {}
        idx2word = {}
        
        for i in range(data_size) :
            tok = data_token[i]
            idx = data_index[i]
            
            word2idx[tok] = idx
            idx2word[idx] = tok
        
        self.word2idx = word2idx
        self.idx2word = idx2word
    
    def encode_sen(self, sen) :
        idx_list = [self.word2idx[tok] if tok in self.word2idx else Token.UNK for tok in sen] 
        idx_list = [Token.SOS] + idx_list + [Token.EOS]
        return idx_list
    
    def encode(self) :
        return [self.encode_sen(sen) for sen in self.tok_data]
