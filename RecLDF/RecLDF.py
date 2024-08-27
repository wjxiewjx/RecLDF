import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec


class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)

    
class MultiFeatureLayer(nn.Module):
    """MoE-enhanced Adaptor
    """
    def __init__(self, n_feat, layers, feature_embedding_size, dropout=0.0, noise=False):
        super(MultiFeatureLayer, self).__init__()

        self.n_feat = n_feat
        self.noisy_gating = noise
        self.input_dim = layers[0]
        self.output_dim = layers[1]
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(layers[0], layers[1])) for _ in range(n_feat)
        ])
        self.act_fun = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.feature_embedding = nn.Parameter(torch.zeros(feature_embedding_size), requires_grad=True)
        self.linear = PWLayer(feature_embedding_size, feature_embedding_size, dropout)
        self.softplus = nn.Softplus()

    def forward(self, x, train_flag=0):
        gate = self.softplus(self.linear(x))
        x = x * gate
        feature_embedding = self.feature_embedding
        assert feature_embedding.shape[0] == x.shape[-1]
        x = x + feature_embedding
        assert x.shape[-1] % self.input_dim == 0
        split_x = torch.split(x, self.input_dim, dim=-1)         
        assert len(split_x) == self.n_feat, "split_x must have n_feat number of matrices"   
        result = 0
        for i in range(self.n_feat):
            result += torch.matmul(split_x[i], self.weights[i])  
        result /= self.n_feat      
        result = self.act_fun(result)
        result = self.layer_norm(result)

        return result
       
class Decoder(nn.Module):
    """MoE-enhanced Adaptor Decoder
    """
    def __init__(self, n_feat, layers, feature_embedding_size, dropout=0.0):
        super(Decoder, self).__init__()

        self.n_feat = n_feat
        self.input_dim = layers[1]
        self.output_dim = layers[0]
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(layers[1], layers[0])) for _ in range(n_feat)
        ])
        self.act_fun = nn.ReLU()
        self.layer_norm = nn.LayerNorm(feature_embedding_size)
        self.feature_embedding = nn.Parameter(torch.zeros(feature_embedding_size), requires_grad=True)
        self.linear = PWLayer(feature_embedding_size, feature_embedding_size, dropout)
        self.linear_layer = nn.Linear(self.input_dim, feature_embedding_size, bias=False)
        self.softplus = nn.Softplus()

    def forward(self, x, train_flag=0):       
        x = self.linear_layer(x)
        
        feature_embedding = self.feature_embedding
        assert feature_embedding.shape[0] == x.shape[-1]
        result = x - feature_embedding

        gate = self.softplus(self.linear(result))
        result = result / gate
        
        result = self.act_fun(result)
        result = self.layer_norm(result)

        return result
 

class RecLDF(SASRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.train_stage = config['train_stage']
        self.temperature = config['temperature']
        self.lam = config['lambda']
        self.plm_size = config['plm_size']

        assert self.train_stage in [
            'pretrain', 'inductive_ft', 'transductive_ft'
        ], f'Unknown train stage: [{self.train_stage}]'

        if self.train_stage in ['pretrain', 'inductive_ft']:
            self.item_embedding = None
            # for `transductive_ft`, `item_embedding` is defined in SASRec base model
        if self.train_stage in ['inductive_ft', 'transductive_ft']:
            # `plm_embedding` in pre-train stage will be carried via dataloader
            self.plm_embedding = copy.deepcopy(dataset.plm_embedding)
        
        self.item_num = dataset.item_num 
        self.mse_loss = nn.MSELoss(reduction='none')
        
        self.multi_feature_adaptor = MultiFeatureLayer(
            config['n_feature'],
            config['adaptor_layers'],
            self.plm_size,
            config['adaptor_dropout_prob']           
        )

        self.decoder = Decoder(
            config['n_feature'],
            config['adaptor_layers'],
            self.plm_size,
            config['adaptor_dropout_prob'] 
        )
        

    def forward(self, item_seq, item_emb, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        if self.train_stage == 'transductive_ft':
            input_emb = input_emb + self.item_embedding(item_seq)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def seq_item_contrastive_task(self, seq_output, same_pos_id, interaction):
        pos_items_emb = self.multi_feature_adaptor(interaction['pos_item_emb'])
        pos_items_emb = F.normalize(pos_items_emb, dim=1)

        pos_logits = (seq_output * pos_items_emb).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    def seq_seq_contrastive_task(self, seq_output, same_pos_id, interaction):
        item_seq_aug = interaction[self.ITEM_SEQ + '_aug']
        item_seq_len_aug = interaction[self.ITEM_SEQ_LEN + '_aug']
        item_emb_list_aug = self.multi_feature_adaptor(interaction['item_emb_list_aug'])
        seq_output_aug = self.forward(item_seq_aug, item_emb_list_aug, item_seq_len_aug)
        seq_output_aug = F.normalize(seq_output_aug, dim=1)

        pos_logits = (seq_output * seq_output_aug).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, seq_output_aug.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    def pretrain(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.multi_feature_adaptor(interaction['item_emb_list'])
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        seq_output = F.normalize(seq_output, dim=1)

        # Remove sequences with the same next item
        pos_id = interaction['item_id']
        same_pos_id = (pos_id.unsqueeze(1) == pos_id.unsqueeze(0))
        same_pos_id = torch.logical_xor(same_pos_id, torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device))

        loss_seq_item = self.seq_item_contrastive_task(seq_output, same_pos_id, interaction)
        loss_seq_seq = self.seq_seq_contrastive_task(seq_output, same_pos_id, interaction)
        loss = loss_seq_item + self.lam * loss_seq_seq
        return loss

    def calculate_loss_supervised(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        item_emb_list = self.multi_feature_adaptor(self.plm_embedding(item_seq), train_flag=1)
        
        supervised = self.decoder(item_emb_list).to(item_emb_list.device)
        batchsize, sequence_length, hidden_size = supervised.shape
        mask = torch.arange(sequence_length, device=item_emb_list.device).expand(batchsize, sequence_length) < item_seq_len.unsqueeze(1)
        mask = mask.unsqueeze(-1).expand(batchsize, sequence_length, hidden_size)
        loss_supervised = self.mse_loss(supervised, self.plm_embedding(item_seq))
        mse_loss = loss_supervised * mask.float()
        mse_loss = mse_loss.sum() / mask.float().sum()

        return mse_loss

    def calculate_loss(self, interaction, lamba=0.2):
        if self.train_stage == 'pretrain':
            return self.pretrain(interaction)
        
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        item_emb_list = self.multi_feature_adaptor(self.plm_embedding(item_seq), train_flag=1)
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)

        test_item_emb = self.multi_feature_adaptor(self.plm_embedding.weight)
        if self.train_stage == 'transductive_ft':
            test_item_emb = test_item_emb + self.item_embedding.weight

        seq_output = F.normalize(seq_output, dim=1)
        test_item_emb = F.normalize(test_item_emb, dim=1)

        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        pos_items = interaction[self.POS_ITEM_ID]
        #self.item_embedding(item_seq)
        pos_item_emb = test_item_emb[pos_items]
        mse_loss = F.mse_loss(seq_output, pos_item_emb)

        loss = self.loss_fct(logits, pos_items) + lamba * mse_loss
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.multi_feature_adaptor(self.plm_embedding(item_seq))
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_items_emb = self.multi_feature_adaptor(self.plm_embedding.weight)
        if self.train_stage == 'transductive_ft':
            test_items_emb = test_items_emb + self.item_embedding.weight

        seq_output = F.normalize(seq_output, dim=-1)
        test_items_emb = F.normalize(test_items_emb, dim=-1)

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
