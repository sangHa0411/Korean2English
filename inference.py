
import copy
import torch
import torch.nn.functional as F
from dataset import Token

class Translator :
    def __init__(self, encoder, decoder, padding_mask, lookahead_mask, max_size=128, beam_size=5) :
        self.encoder = encoder
        self.decoder = decoder
        self.padding_mask = padding_mask
        self.lookahead_mask = lookahead_mask
        self.max_size = max_size
        self.beam_size = beam_size

    def search(self, out_tensor, idx) :
        idx_tensor = out_tensor[0][idx]
        prob_tensor = F.softmax(idx_tensor, dim=0)
        arg_tensor = torch.argsort(prob_tensor, dim=0)

        arg_list = arg_tensor.detach().cpu().numpy()[:self.beam_size]
        prob_list = []
        for arg in arg_list :
            prob = prob_tensor[arg]
            prob_list.append(prob)
        return arg_list, prob_list

    def select(self, ids_data) :
        ids_sorted = sorted(ids_data , key = lambda x : x[1] , reverse=True)
        ids_selected = ids_sorted[:self.beam_size]
        return ids_selected

    def translate(self, src_id_tensor, src_mask_tensor) :
        src_feature_tensor = self.encoder(src_id_tensor, src_mask_tensor)

        idx = 0
        tar_ids_data = [([Token.SOS], 1.0)]
        while idx < self.max_size :
            end_flag = True
            tar_ids_next_data = []

        for i in range(len(tar_ids_data)) :
            tar_ids = tar_ids_data[i]
            tar_id_list, tar_id_prob = tar_ids

            if tar_id_list[-1] == Token.EOS :
                continue

            end_flag = False
            padding_size = self.max_size - len(tar_id_list)
            tar_id_tensor = torch.tensor(tar_id_list + [0] * padding_size).unsqueeze(0).to(device)
            tar_mask_tensor = self.padding_mask(tar_id_tensor)
            tar_mask_tensor = self.lookahead_mask(tar_mask_tensor)

            tar_out_tensor = self.decoder(tar_id_tensor, tar_mask_tensor, src_feature_tensor, src_mask_tensor)
            arg_list, prob_list = self.search(tar_out_tensor, idx)

            for i in range(self.beam_size) :
                arg = arg_list[i]
                prob = prob_list[i].detach().cpu().numpy()

                tar_id_next = copy.deepcopy(tar_id_list)
                tar_id_next.append(arg)
                tar_ids_next_data.append((tar_id_next, (tar_id_prob * prob)))

            if end_flag == True :
                break

            idx += 1
            tar_ids_data = self.select(tar_ids_next_data)

        tar_id_list = tar_ids_data[0]
        return tar_id_list