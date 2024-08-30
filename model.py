import torch.nn as nn
import torch
from transformers import BertModel
from transformers import BertForTokenClassification
from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer, AdamW, get_cosine_schedule_with_warmup
import torch.nn.functional as F

from kan_convolutional.KANConv import KAN_Convolutional_Layer
class BertForIntentClassificationAndSlotFilling(nn.Module):
    def __init__(self, config):
        super(BertForIntentClassificationAndSlotFilling, self).__init__()
        self.config = XLMRobertaConfig.from_pretrained(config.bert_dir)
        self.bert = XLMRobertaModel.from_pretrained(config.bert_dir)
        self.bert_config = self.bert.config
        self.sequence_classification = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(self.config.hidden_size, config.seq_num_labels),
        )
        self.token_classification = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(self.config.hidden_size, config.token_num_labels),
        )
        self.domain_classification = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(self.config.hidden_size, config.domain_num_labels),
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=1024,
            nhead=8,
            dim_feedforward=1024,  # 前馈神经网络中的隐藏层维度
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        num_layers = 6
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.lstm_layer = nn.LSTM(
            input_size=self.config.hidden_size,
            hidden_size=int(self.config.hidden_size/2),
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.conv1 = nn.Conv1d(in_channels=self.config.hidden_size, out_channels=self.config.hidden_size, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=self.config.hidden_size, out_channels=self.config.hidden_size, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=self.config.hidden_size, out_channels=self.config.hidden_size, kernel_size=5, padding=2)

        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=self.config.hidden_size*3, nhead=8, dropout=0.1)
        self.slot_decoder = nn.TransformerDecoderLayer(d_model=self.config.hidden_size, nhead=8, dropout=0.1)
        self.linear_layer1 = nn.Linear(3072, 2048)
        self.linear_layer2 = nn.Linear(2048, 1024)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                ):
        bert_output = self.bert(input_ids, attention_mask)
        pooler_output = bert_output[1]
        slot_output = bert_output[0]

        # x = token_output.permute(0, 2, 1)
        # x1 = self.conv1(x)
        # x3 = self.conv3(x)
        # x5 = self.conv5(x)
        # x = torch.cat([x1, x3, x5], dim=1).permute(0, 2, 1)
        #
        # # Transformer Encoder
        # x = self.transformer_encoder(x)
        # x = self.linear_layer1(x)
        # x = self.linear_layer2(x)
        # # Slot-Filling Decoder
        # slot_output = self.slot_decoder(x, x, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
        #                                 memory_key_padding_mask=None)

        token_output = self.encoder(slot_output)
        token_output,_ = self.lstm_layer(token_output)




        seq_output = self.sequence_classification(pooler_output)
        token_output = self.token_classification(token_output)
        domain_output = self.domain_classification(pooler_output)
        return seq_output, token_output, domain_output


