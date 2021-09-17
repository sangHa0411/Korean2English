import os
import sys
import random
import argparse
import multiprocessing
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import TransformerDecoder, TransformerEncoder
from dataset import *
from preprocessor import *
from collator import *

def schedule_fn(epoch, d_model, init_lr, warmup_steps) :
    step_num = epoch + 1
    val1 = d_model ** (-0.5)
    arg1 = step_num ** (-0.5)
    arg2 = (warmup_steps ** (-1.5)) * step_num
    val2 = min(arg1 , arg2) 
    return (val1 * val2) / init_lr

def progressLearning(value, endvalue, loss, acc, bar_length=50):
    percent = float(value + 1) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\r[{0}] {1}/{2} \t Loss : {3:.3f} , Acc : {4:.3f}".format(arrow + spaces, value+1, endvalue, loss, acc))
    sys.stdout.flush()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(args) :
    # -- Seed
    seed_everything(args.seed)

    # -- Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- Data
    text_data = pd.read_csv(args.data_dir)
    kor_data = list(text_data['원문'])
    en_data = list(text_data['번역문'])

    # -- Tokenizer & Encoder
    en_text_path = os.path.join(args.token_dir, 'english.txt')
    if os.path.exists(en_text_path) == False:
        write_data(en_data, en_text_path, preprocess_en)
    en_spm = get_spm(args.token_dir, 'english.txt' , 'en_spm' , args.token_size)
    en_v_size = len(en_spm)

    kor_text_path = os.path.join(args.token_dir, 'korean.txt')
    if os.path.exists(kor_text_path) == False:
        write_data(kor_data, kor_text_path, preprocess_kor)
    kor_spm = get_spm(args.token_dir, 'korean.txt' , 'kor_spm' , args.token_size)
    kor_v_size = len(kor_spm)

    # -- Dataset
    data_size = len(text_data)

    en_index_data = []
    kor_index_data = []
    for i in range(data_size) :
        en_sen = en_data[i]
        en_sen = preprocess_en(en_sen)
        en_index_list = en_spm.encode_as_ids(en_sen)
        en_index_data.append(en_index_list)
        kor_sen = kor_data[i]
        kor_sen = preprocess_kor(kor_sen)
        kor_index_list = kor_spm.encode_as_ids(kor_sen)
        kor_index_data.append(kor_index_list)


    dset = TranslationDataset(kor_index_data, en_index_data, args.max_size)
    train_dset, val_dset = dset.split()

    train_len = [(len(data[0]),len(data[1])) for data in train_dset]
    val_len = [(len(data[0]),len(data[1])) for data in val_dset]

    # -- DataLoader
    train_collator = Collator(train_len, args.batch_size)
    train_loader = DataLoader(train_dset,
        num_workers=multiprocessing.cpu_count()//2,
        batch_sampler=train_collator.sample(),
        collate_fn=train_collator
    )
    val_collator = Collator(val_len, args.val_batch_size)
    val_loader = DataLoader(val_dset,
        num_workers=multiprocessing.cpu_count()//2,
        batch_sampler=val_collator.sample(),
        collate_fn=val_collator
    )

    # -- Model
    # Transformer Encoder
    encoder = TransformerEncoder(layer_size=args.layer_size, 
        max_size=args.max_size, 
        v_size=kor_v_size, 
        d_model=args.embedding_size,
        num_heads=args.head_size,
        hidden_size=args.hidden_size,
        drop_rate=0.1,
        norm_rate=1e-6,
        cuda_flag=use_cuda
    )
    # Transformer Decoder
    decoder = TransformerDecoder(layer_size=args.layer_size, 
        max_size=args.max_size, 
        v_size=en_v_size, 
        d_model=args.embedding_size,
        num_heads=args.head_size,
        hidden_size=args.hidden_size,
        drop_rate=0.1,
        norm_rate=1e-6,
        cuda_flag=use_cuda
    )

    # Set Embedding
    kor_embedding_path = os.path.join(args.embedding_dir,'kor_weight.npy')
    if os.path.exists(kor_embedding_path) :
        print('Load Encoder Embedding')
        kor_weight = np.load(kor_embedding_path)
        encoder.set_embedding(kor_weight)
    encoder = encoder.to(device)

    en_embedding_path = os.path.join(args.embedding_dir,'en_weight.npy')
    en_bias_path = os.path.join(args.embedding_dir,'en_bias.npy')
    if os.path.exists(en_embedding_path) :
        print('Load Decoder Embedding')
        en_weight = np.load(en_embedding_path)
        decoder.set_embedding(en_weight)
    if os.path.exists(en_bias_path) :
        print('Load Decoder Bias')
        en_bias = np.load(en_bias_path)
        decoder.set_bias(en_bias)
    decoder = decoder.to(device)

    model_parameters = chain(encoder.parameters(), decoder.parameters())
    
    # -- Optimizer
    optimizer = optim.Adam(model_parameters, lr=1e-4, betas=(0.9,0.98), eps=1e-9)

    # -- Scheduler
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, 
        lr_lambda = lambda epoch: schedule_fn(epoch = epoch,
            d_model = args.embedding_size, 
            init_lr = 1e-4, 
            warmup_steps=args.warmup_steps)
    )
    
    # -- Logging
    writer = SummaryWriter(args.log_dir)

    # -- Criterion 
    criterion = nn.CrossEntropyLoss().to(device)

    # -- Training
    min_loss = np.inf
    stop_count = 0
    log_count = 0
    # for each epoch
    for epoch in range(args.epochs) :
        idx = 0
        encoder.train()
        decoder.train()
        print('Epoch : %d/%d \t Learning Rate : %e' %(epoch, args.epochs, optimizer.param_groups[0]["lr"]))
        # training process
        for data in train_loader :
            en_in = data['encoder_in'].long().to(device)
            de_in = data['decoder_in'].long().to(device)
        
            de_label = data['decoder_out'].long().to(device)
            de_label = torch.reshape(de_label, (-1,))

            optimizer.zero_grad()
        
            en_output, en_pad = encoder(en_in)
            de_output = decoder(de_in, en_output, en_pad)
            de_output = torch.reshape(de_output, (-1,en_v_size))

            loss = criterion(de_output , de_label)
            acc = (torch.argmax(de_output, dim=-1) == de_label).float().mean()

            loss.backward()
            optimizer.step()
        
            progressLearning(idx, len(train_loader), loss.item(), acc.item())

            if (idx + 1) % 10 == 0 :
                writer.add_scalar('train/loss', loss.item(), log_count)
                writer.add_scalar('train/acc', acc.item(), log_count)
                log_count += 1
            idx += 1

        # validation process
        with torch.no_grad() :
            encoder.eval()
            decoder.eval()

            loss_eval = 0.0
            acc_eval = 0.0

            for data in val_loader :
                en_in = data['encoder_in'].long().to(device)
                de_in = data['decoder_in'].long().to(device)
        
                de_label = data['decoder_out'].long().to(device)
                de_label = torch.reshape(de_label, (-1,))

                en_output, en_pad = encoder(en_in)
                de_output = decoder(de_in, en_output, en_pad)
                de_output = torch.reshape(de_output, (-1,en_v_size))

                loss_eval += criterion(de_output, de_label)
                acc_eval += (torch.argmax(de_output, dim=-1) == de_label).float().mean()  

            loss_eval /= len(val_loader)
            acc_eval /= len(val_loader)

        writer.add_scalar('test/loss', loss_eval.item(), epoch)
        writer.add_scalar('test/acc', acc_eval.item(), epoch)
    
        if loss_eval < min_loss :
            min_loss = loss_eval
            torch.save({'epoch' : (epoch) ,  
                'encoder_state_dict' : encoder.state_dict() , 
                'decoder_state_dict' : decoder.state_dict() , 
                'loss' : loss_eval.item() , 
                'acc' : acc_eval.item()} , 
                f'./Model/checkpoint_transformer.pt')        
            stop_count = 0 
        else :
            stop_count += 1
            if stop_count >= 5 :      
                print('\tTraining Early Stopped')
                break
            
        scheduler.step()
        print('\nVal Loss : %.3f \t Val Accuracy : %.3f\n' %(loss_eval, acc_eval))
    
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=777, help='random seed (default: 777)')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
    parser.add_argument('--token_size', type=int, default=7000, help='merge size of bpe (default: 7000)')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='warmup steps of train (default: 4000)')
    parser.add_argument('--max_size', type=int, default=30, help='max size of sequence (default: 30)')
    parser.add_argument('--layer_size', type=int, default=6, help='layer size of model (default: 6)')
    parser.add_argument('--embedding_size', type=int, default=512, help='embedding size of token (default: 512)')
    parser.add_argument('--hidden_size', type=int, default=2048, help='hidden size of position-wise layer (default: 2048)')
    parser.add_argument('--head_size', type=int, default=8, help='head size of multi head attention (default: 8)')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=128, help='input batch size for validing (default: 128)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default='../Data/korean_dialogue_translation.csv')
    parser.add_argument('--model_dir', type=str, default='./Model')
    parser.add_argument('--token_dir', type=str, default='./Token')
    parser.add_argument('--embedding_dir', type=str, default='./Embedding')
    parser.add_argument('--log_dir' , type=str , default='./Log')

    args = parser.parse_args()
    train(args)

