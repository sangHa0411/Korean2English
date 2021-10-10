import re
import os
import sentencepiece as spm
from dataset import Token

spm_templates= '--input={} \
--pad_id={} \
--bos_id={} \
--eos_id={} \
--unk_id={} \
--model_prefix={} \
--vocab_size={} \
--character_coverage={} \
--model_type={}'

def preprocess(sen) :
    sen = re.sub('[^a-zA-Z가-힣0-9.,!?:;\'\" ]', '', sen)
    sen = re.sub(' {2,}' , ' ' , sen)
    return sen

def write_data(text_list, text_path, preprocessor) :
    with open(text_path, 'w') as f :
        for sen in text_list :
            sen = preprocessor(sen)
            f.write(sen + '\n')

def train_spm(dir_path, text_name, model_name, vocab_size) :
    text_path = os.path.join(dir_path, text_name)
    model_path = os.path.join(dir_path, model_name)
    spm_cmd = spm_templates.format(text_path, 
        Token.PAD,
        Token.SOS, 
        Token.EOS, 
        Token.UNK, 
        model_path, 
        vocab_size, 
        1.0, 
        'bpe')
    spm.SentencePieceTrainer.Train(spm_cmd)

def get_spm(dir_path, model_name) :
    model_path = os.path.join(dir_path, model_name) + '.model'
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    sp.SetEncodeExtraOptions('bos:eos')
    return sp
