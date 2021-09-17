import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PaddingMask(nn.Module) :
    def __init__(self) :
        super(PaddingMask , self).__init__() 
    
    def forward(self, in_tensor) :
        batch_size, seq_size = in_tensor.shape
        flag_tensor = torch.where(in_tensor == 0.0 , 1.0 , 0.0)
        flag_tensor = torch.reshape(flag_tensor , (batch_size, 1, 1, seq_size)) 
        return flag_tensor

class LookAheadMask(nn.Module) :
    def __init__(self, cuda_flag) :
        super(LookAheadMask, self).__init__() 
        self.cuda_flag = cuda_flag
        
    def get_mask(self, sen_size) :
        # masking tensor
        mask_array = 1 - np.tril(np.ones((sen_size,sen_size)) , 0)
        mask_tensor = torch.tensor(mask_array , dtype = torch.float32 , requires_grad=False)
        mask_tensor = mask_tensor.unsqueeze(0) # shape : (1, sen_size, sen_size)
        return mask_tensor
    
    def forward(self, in_tensor) :
        batch_size, _, _, seq_size = in_tensor.shape
        mask_tensor = self.get_mask(seq_size)
        if self.cuda_flag :
            mask_tensor = mask_tensor.cuda() 
        lookahead_mask_tensor = torch.maximum(in_tensor, mask_tensor)
        return lookahead_mask_tensor

class PositionalEncoding(nn.Module) :
    def __init__(self, max_len, d_model, cuda_flag) :
        super(PositionalEncoding , self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.cuda_flag = cuda_flag
        # w : weight
        # pe : Encoding tensor
        self.w = torch.sqrt(torch.tensor(d_model, dtype=torch.float32, requires_grad=False))
        self.pe = self.get_embedding(max_len, d_model)
        if cuda_flag == True :
            self.w = self.w.cuda()
            self.pe = self.pe.cuda()
        
    # Embedding tensor : (batch_size, sen_size, embedding_dim)
    # Making Encoding tensor (1, sen_size, embedding_dim)
    def get_embedding(self, pos_len, d_model) :
        pos_vec = torch.arange(pos_len).float()
        pos_vec = pos_vec.unsqueeze(1)

        i_vec = torch.arange(d_model).float() / 2
        i_vec = torch.floor(i_vec) * 2
        i_vec = i_vec.unsqueeze(0) / d_model
        i_vec = 1 / torch.pow(1e+4 , i_vec)

        em = torch.mul(pos_vec, i_vec)
        pe = torch.zeros(pos_len, d_model, requires_grad=False)
        sin_em = torch.sin(em)
        cos_em = torch.cos(em)

        pe[:,::2] = sin_em[:,::2]
        pe[:,1::2] = cos_em[:,1::2]

        return pe.unsqueeze(0)

    # input tensor : (batch_size, sen_size, embedding_dim)
    def forward(self, in_tensor) :
        batch_size, seq_size, em_dim = in_tensor.shape                  
        en_tensor = (in_tensor * self.w) + self.pe[:,:seq_size,:]
        return en_tensor

# Multihead Attention Layer
class MultiHeadAttention(nn.Module) :
    def __init__(self, d_model, num_heads) :
        super(MultiHeadAttention , self).__init__()
        self.d_model = d_model # vector size 
        self.num_heads = num_heads # head_size
        self.depth = int(d_model / num_heads)

        self.q_layer = nn.Linear(d_model, d_model)
        self.k_layer = nn.Linear(d_model, d_model)
        self.v_layer = nn.Linear(d_model, d_model)
        self.o_layer = nn.Linear(d_model, d_model)

        self.scale = torch.sqrt(torch.tensor(self.depth , dtype=torch.float32 , requires_grad=False))

    # tensor shape : (batch_size, sen_size, d_model)
    def split(self, tensor) :
        sen_size = tensor.shape[1]
        tensor = torch.reshape(tensor, (-1, sen_size, self.num_heads, self.depth))     
        tensor = tensor.permute(0,2,1,3) # (batch_size, num_heads, sen_size, depth)
        return tensor

    # tensor shape : (batch_size, num_heads, sen_size, dk)
    def merge(self, tensor) :
        sen_size = tensor.shape[2]
        tensor = tensor.permute(0,2,1,3) # (batch_size, sen_size, num_heads, depth)
        tensor = torch.reshape(tensor, (-1, sen_size, self.d_model)) #(batch_size, sen_size, embedding_dim)
        return tensor
    
    # scaled dot production
    def scaled_dot_production(self, q_tensor, k_tensor, v_tensor, m_tensor) :
        q_tensor = self.split(q_tensor)
        k_tensor = self.split(k_tensor)
        v_tensor = self.split(v_tensor)
        
        k_tensor_T = k_tensor.permute(0,1,3,2) # (batch_size, num_heads, depth, sen_size)

        qk_tensor = torch.matmul(q_tensor , k_tensor_T) # (batch_size, num_heads, sen_size, sen_size)
        qk_tensor /= self.scale

        if m_tensor != None :
            qk_tensor -= (m_tensor*1e+6)
            
        qk_tensor = F.softmax(qk_tensor, dim=-1)
        att = torch.matmul(qk_tensor, v_tensor) # (batch_size, num_heads, sen_size, depth)

        return att

    def forward(self, q_in, k_in, v_in, m_in) :
        q_tensor = self.q_layer(q_in)
        k_tensor = self.k_layer(k_in)
        v_tensor = self.v_layer(v_in)
        
        att_tensor = self.scaled_dot_production(q_tensor , k_tensor , v_tensor , m_in)
        multi_att_tensor = self.merge(att_tensor)
        
        o_tensor = self.o_layer(multi_att_tensor)
        return o_tensor

# Feedforward layer
class FeedForward(nn.Module) :
    def __init__(self, hidden_size, d_model) :
        super(FeedForward , self).__init__()
        self.hidden_size = hidden_size
        self.d_model = d_model
        # relu activation and input, output dim are same
        self.ff = nn.Sequential(nn.Linear(d_model , hidden_size), 
                                nn.ReLU(),
                                nn.Linear(hidden_size , d_model))

    def forward(self , in_tensor) :
        o_tensor = self.ff(in_tensor)
        return o_tensor

# Transformer Encoder Block
class EncoderBlock(nn.Module) :
    def __init__(self, d_model, num_heads, hidden_size, drop_rate, norm_rate) :
        super(EncoderBlock, self).__init__()
        # multihead attention layer & feedforward layer
        self.mha_layer = MultiHeadAttention(d_model , num_heads)
        self.ff_layer = FeedForward(hidden_size , d_model)
        # dropout layer & layer normalization layer
        self.drop1_layer = nn.Dropout(drop_rate)
        self.norm2_layer = nn.LayerNorm(d_model, eps=norm_rate)
        self.drop1_layer = nn.Dropout(drop_rate)
        self.norm2_layer = nn.LayerNorm(d_model, eps=norm_rate)
      
        self.init_param()
                
    # He Initialization
    def init_param(self) :
        gain = 2 ** (1/2)
        for m in self.modules() :
            if isinstance(m , nn.Linear) :
                nn.init.kaiming_normal_(m.weight , gain)
                nn.init.zeros_(m.bias)
        
    def forward(self, in_tensor, m_tensor) :
        # mutlihead attention sub layer
        mha_tensor = self.mha_layer(in_tensor , in_tensor , in_tensor , m_tensor)
        mha_tensor = self.drop_layer1(mha_tensor)
        h_tensor = self.norm_layer1(in_tensor + mha_tensor)
        # feed forward sub layer
        ff_tensor = self.ff_layer(h_tensor)
        ff_tensor = self.drop_layer2(ff_tensor)
        o_tensor = self.norm_layer2(h_tensor + ff_tensor)

        return o_tensor

# Transformer Decoder Block
class DecoderBlock(nn.Module) :
    def __init__(self, d_model, num_heads, hidden_size, drop_rate, norm_rate) :
        super(DecoderBlock, self).__init__()
        # multihead attention layer & feedforward layer
        self.m_mha_layer = MultiHeadAttention(d_model , num_heads)
        self.mha_layer = MultiHeadAttention(d_model , num_heads)
        self.ff_layer = FeedForward(hidden_size , d_model)
        # dropout layer & layer normalization layer
        self.drop1_layer = nn.Dropout(drop_rate)
        self.norm1_layer = nn.LayerNorm(d_model, eps=norm_rate)
        self.drop2_layer = nn.Dropout(drop_rate)
        self.norm2_layer = nn.LayerNorm(d_model, eps=norm_rate)
        self.drop3_layer = nn.Dropout(drop_rate)
        self.norm3_layer = nn.LayerNorm(d_model, eps=norm_rate)

        self.init_param()
                
    # He Initialization
    def init_param(self) :
        gain = 2 ** (1/2)
        for m in self.modules() :
            if isinstance(m , nn.Linear) :
                nn.init.kaiming_normal_(m.weight , gain)
                nn.init.zeros_(m.bias)
        
    def forward(self, in_tensor, en_out_tensor, en_pad_mask, de_lookahead_mask) :
        # masked multihead attention sub layer
        # query : in_tensor, key : in_tensor, value : in_tensor, mask ; look ahead mask
        m_mha_tensor = self.m_mha_layer(in_tensor, in_tensor, in_tensor, de_lookahead_mask)
        m_mha_tensor = self.drop1_layer(m_mha_tensor)
        h_tensor = self.norm1_layer(in_tensor + m_mha_tensor)
        # multihead attention sub layer
        # query : decoder output, key : encoder output, value : encoder output , mask ; pad mask
        mha_tensor = self.mha_layer(h_tensor, en_out_tensor, en_out_tensor, en_pad_mask)
        mha_tensor = self.drop2_layer(mha_tensor)
        a_tensor = self.norm2_layer(h_tensor+mha_tensor)
        # feedforward sub layer
        ff_tensor = self.ff_layer(a_tensor)
        ff_tensor = self.drop3_layer(ff_tensor)
        o_tensor = self.norm3_layer(a_tensor+ff_tensor)
        
        return o_tensor

# Transformer Encoder
class TransformerEncoder(nn.Module) :
    def __init__(self, layer_size, max_size, v_size, d_model, num_heads, hidden_size, 
                 drop_rate, norm_rate, cuda_flag) :
        super(TransformerEncoder , self).__init__()
        self.layer_size = layer_size
        self.max_size = max_size
        self.v_size = v_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.drop_rate = drop_rate
        self.norm_rate = norm_rate

        self.em = nn.Embedding(num_embeddings=v_size, 
                               embedding_dim=d_model, 
                               padding_idx=0) # embedding

        self.pos = PositionalEncoding(max_size, d_model, cuda_flag) # positional encoding
        self.pad = PaddingMask() # masking
        self.en_blocks = nn.ModuleList()
        self.drop_layer = nn.Dropout(drop_rate)
        self.norm_layer = nn.LayerNorm(d_model , eps=norm_rate)

        for i in range(layer_size) :
            self.en_blocks.append(EncoderBlock(d_model, num_heads, hidden_size, drop_rate, norm_rate))
        
        self.init_param()
            
    def set_embedding(self, em_weight) :
        em_v_size, em_h_size = em_weight.shape
        assert em_v_size == self.v_size
        assert em_h_size == self.d_model
        em_weight = torch.tensor(em_weight)
        self.em = nn.Embedding.from_pretrained(em_weight, freeze=True, padding_idx=0)
            
    def init_param(self) :
        nn.init.normal_(self.em.weight, mean=0.0, std=0.1)
            
    def forward(self, in_tensor) :
        # masking tensor
        en_pad_mask = self.pad(in_tensor)
        # encoder input tensor
        em_tensor = self.em(in_tensor) # embedding
        en_tensor = self.pos(em_tensor) # positional encoding
        en_tensor = self.drop_layer(en_tensor) # dropout layer
        
        tensor_ptr = en_tensor
        for i in range(self.layer_size) :
            tensor_ptr = self.en_blocks[i](tensor_ptr, en_pad_mask)
        
        return tensor_ptr, en_pad_mask

# Transformer Decoder
class TransformerDecoder(nn.Module) :
    def __init__(self, layer_size, max_size, v_size, d_model, num_heads, hidden_size, 
                 drop_rate, norm_rate, cuda_flag) :
        super(TransformerDecoder , self).__init__()
        self.layer_size = layer_size
        self.max_size = max_size
        self.v_size = v_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.drop_rate = drop_rate
        self.norm_rate = norm_rate
        
        self.em = nn.Embedding(num_embeddings=v_size, embedding_dim=d_model, padding_idx=0) # embedding
        self.pos = PositionalEncoding(max_size, d_model, cuda_flag) # positional encoding
        self.pad = PaddingMask() # padding masking
        self.lookahead = LookAheadMask(cuda_flag) # lookahead masking
        
        self.de_blocks = nn.ModuleList()

        self.drop_layer = nn.Dropout(drop_rate)
        self.norm_layer = nn.LayerNorm(d_model , eps=norm_rate)
        
        self.o_layer = nn.Linear(d_model, v_size)

        for i in range(layer_size) :
            self.de_blocks.append(DecoderBlock(d_model, num_heads, hidden_size, drop_rate, norm_rate))
    
        self.init_param()
        
    def init_param(self) :
        nn.init.normal_(self.em.weight, mean=0.0, std=0.1)
        nn.init.xavier_normal_(self.o_layer.weight, gain=1.0)
        nn.init.zeros_(self.o_layer.bias)
        
    # embedding layer, output layer weight from pretrained : not trainable
    def set_embedding(self, em_weight) :
        em_v_size, em_h_size = em_weight.shape
        assert em_v_size == self.v_size
        assert em_h_size == self.d_model
        em_weight = torch.tensor(em_weight)
        self.em = nn.Embedding.from_pretrained(em_weight, freeze=True, padding_idx=0)
        self.o_layer.weight = nn.Parameter(em_weight, requires_grad=False)
        
    # output layer bias from pretrained : not trainable
    def set_bias(self, bias) :
        bias = torch.tensor(bias)
        self.o_layer.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, in_tensor, en_out_tensor, en_pad_mask) :
        # masking tensor
        de_pad_mask = self.pad(in_tensor)
        de_lookahead_mask = self.lookahead(de_pad_mask)
        # decoder input tensor
        em_tensor = self.em(in_tensor) # embedding
        de_tensor = self.pos(em_tensor) # positional encoding
        de_tensor = self.drop_layer(de_tensor) # dropout layer
        
        tensor_ptr = de_tensor
        for i in range(self.layer_size) :
            tensor_ptr = self.de_blocks[i](tensor_ptr, en_out_tensor, en_pad_mask, de_lookahead_mask)
        o_tensor = self.o_layer(tensor_ptr)
        
        return o_tensor

