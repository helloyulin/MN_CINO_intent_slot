from collections import Counter

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertTokenizer
from torch.optim import Adam
import torch.nn as nn
import torch
import numpy as np
import os
from seqeval.metrics.sequence_labeling import get_entities
import sys
import json
# sys.stdout.reconfigure(encoding='utf-8')
from config import Args
from model import BertForIntentClassificationAndSlotFilling
from dataset import BertDataset
from preprocess import Processor, get_features
import datetime
from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer, AdamW, get_cosine_schedule_with_warmup

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=config.lr)
        self.epoch = args.epoch
        self.device =args.device

    def train(self, train_loader):
        global_step = 0
        total_step = len(train_loader) * self.epoch
        loss_flog = 100;
        # 获取当前时间
        current_time = datetime.datetime.now()
        # 将当前时间格式化为字符串
        current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        # 打印当前时间
        print("开始时间:", current_time_str)
        model.to( self.device)
        print( self.device)
        for epoch in range(self.epoch):
            for step, train_batch in enumerate(train_loader):
                for key in train_batch.keys():
                    train_batch[key] = train_batch[key].to(self.device)
                input_ids = train_batch['input_ids']
                attention_mask = train_batch['attention_mask']
                token_type_ids = train_batch['token_type_ids']
                seq_label_ids = train_batch['seq_label_ids']
                domain_label_ids = train_batch['domain_label_ids']
                token_label_ids = train_batch['token_label_ids']
                seq_output, token_output, domain_output = self.model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                )

                active_loss = attention_mask.view(-1) == 1
                active_logits = token_output.view(-1, token_output.shape[2])[active_loss]
                active_labels = token_label_ids.view(-1)[active_loss]

                seq_loss = self.criterion(seq_output, seq_label_ids)
                domain_loss = self.criterion(domain_output, domain_label_ids)
                token_loss = self.criterion(active_logits, active_labels)
                loss = seq_loss + token_loss + domain_loss
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(f'[train] epoch:{epoch+1} {global_step}/{total_step} loss:{loss.item()}')
                global_step += 1
                # 获取当前时间
            current_time = datetime.datetime.now()
            # 将当前时间格式化为字符串
            current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
            # 打印当前时间
            print("结束时间:", current_time_str)
            print("loss:",loss)
            print("loss_flog:",loss_flog)
            if self.config.do_save and loss_flog>loss:
                loss_flog = loss
                print("saving model...",self.config.save_dir+self.config.save_name)
                self.save(self.config.save_dir, self.config.save_name)

    def test(self, test_loader):
        self.model.eval()
        seq_preds = []
        seq_trues = []
        token_preds = []
        token_trues = []
        with torch.no_grad():
            for step, test_batch in enumerate(test_loader):
                for key in test_batch.keys():
                    test_batch[key] = test_batch[key].to(self.device)
                input_ids = test_batch['input_ids']
                attention_mask = test_batch['attention_mask']
                token_type_ids = test_batch['token_type_ids']
                seq_label_ids = test_batch['seq_label_ids']
                token_label_ids = test_batch['token_label_ids']
                seq_output, token_output, domain_output = self.model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                )
                seq_output = seq_output.detach().cpu().numpy()
                seq_output = np.argmax(seq_output, -1)
                seq_label_ids = seq_label_ids.detach().cpu().numpy()
                seq_label_ids = seq_label_ids.reshape(-1)
                seq_preds.extend(seq_output)
                seq_trues.extend(seq_label_ids)

                token_output = token_output.detach().cpu().numpy()
                token_label_ids = token_label_ids.detach().cpu().numpy()
                token_output = np.argmax(token_output, -1)
                active_len =  torch.sum(attention_mask, -1).view(-1)
                for length, t_output, t_label in zip(active_len, token_output, token_label_ids):
                    t_output = t_output[1:length-1]
                    t_label = t_label[1:length-1]
                    t_ouput = [self.config.id2nerlabel[i] for i in t_output]
                    t_label = [self.config.id2nerlabel[i] for i in t_label]
                    token_preds.append(t_ouput)
                    token_trues.append(t_label)

        acc, precision, recall, f1 = self.get_metrices(seq_trues, seq_preds, 'cls')
        report = self.get_report(seq_trues, seq_preds, 'cls')
        ner_acc, ner_precision, ner_recall, ner_f1 = self.get_metrices(token_trues, token_preds, 'ner')
        ner_report = self.get_report(token_trues, token_preds, 'ner')
        print('意图识别：\naccuracy:{}\nprecision:{}\nrecall:{}\nf1:{}'.format(
            acc, precision, recall, f1
        ))
        print(report)


        print('槽位填充：\naccuracy:{}\nprecision:{}\nrecall:{}\nf1:{}'.format(
            ner_acc, ner_precision, ner_recall, ner_f1
        ))
        print(ner_report)


    def get_metrices(self, trues, preds, mode):
        if mode == 'cls':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            acc = accuracy_score(trues, preds)
            # precision = precision_score(trues, preds, average='micro')
            # recall = recall_score(trues, preds, average='micro')
            # f1 = f1_score(trues, preds, average='micro')
            precision = precision_score(trues, preds,average='weighted')
            recall = recall_score(trues, preds,average='weighted')
            f1 = f1_score(trues, preds,average='weighted')
        elif mode == 'ner':
            from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
            acc = accuracy_score(trues, preds)
            precision = precision_score(trues, preds)
            recall = recall_score(trues, preds)
            f1 = f1_score(trues, preds)
        return acc, precision, recall, f1

    def get_report(self, trues, preds, mode):
        if mode == 'cls':
            from sklearn.metrics import classification_report
            report = classification_report(trues, preds)
        elif mode == 'ner':
            from seqeval.metrics import classification_report
            report = classification_report(trues, preds)
        return report

    def get_report1(self, trues, preds, mode):
        if mode == 'cls':
            from sklearn.metrics import classification_report, accuracy_score

            # 计算准确率
            acc = accuracy_score(trues, preds)

            # 生成分类报告
            report = classification_report(trues, preds)

            # 添加准确率到报告中
            report_with_acc = f"Accuracy: {acc:.2f}\n\n{report}"

        elif mode == 'ner':
            from seqeval.metrics import classification_report, accuracy_score

            # 计算准确率
            acc = accuracy_score(trues, preds)

            # 生成分类报告
            report = classification_report(trues, preds)

            # 添加准确率到报告中
            report_with_acc = f"Accuracy: {acc:.2f}\n\n{report}"

        return report_with_acc
    def save(self, save_path, save_name):
        torch.save(self.model.state_dict(), os.path.join(save_path, save_name))

    def predict(self, text):
        self.model.eval()
        with torch.no_grad():
            tmp_text = [i for i in text]
            inputs = tokenizer.encode_plus(
                text=tmp_text,
                max_length=self.config.max_len,
                padding='max_length',
                truncation='only_first',
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
            )
            for key in inputs.keys():
                inputs[key] = inputs[key].to(self.device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            token_type_ids = inputs['token_type_ids']
            seq_output, token_output, domain_output = self.model(
                input_ids,
                attention_mask,
                token_type_ids,
            )
            seq_output = seq_output.detach().cpu().numpy()
            domain_output = domain_output.detach().cpu().numpy()
            print("max domain_output:",np.max(domain_output))
            token_output = token_output.detach().cpu().numpy()
            seq_output  = np.argmax(seq_output, -1)
            domain_output = np.argmax(domain_output, -1)
            token_output = np.argmax(token_output, -1)
            # print(seq_output, token_output)
            seq_output = seq_output[0]
            domain_output = domain_output[0]
            token_output = token_output[0][1:len(text)+1]
            token_output = [self.config.id2nerlabel[i] for i in token_output]
            print("text:",text)
            print('input_ids:',input_ids)
            print('意图：', self.config.id2seqlabel[seq_output])
            print('domain:',self.config.id2domainlabel[domain_output])
            # print('槽位：', ((i[0],text[i[1]:i[2]+1]) for i in get_entities(token_output)))
            print("槽位：")

            data = {}
            for i in get_entities(token_output):
                data[i[0]] = extract_content_by_position(text,i[1],i[2] + 1)
                print(i[0], text[i[1]:i[2] + 1])
                json_string = json.dumps(data, ensure_ascii=False)
                print(json_string)


import re


def extract_content_by_position(text, start_idx, end_idx):
    """
    根据给定的起始和结束索引，返回句子中相应的内容。句子按空格分割。

    参数:
    - text (str): 输入的文本。
    - start_idx (int): 起始单词的索引。
    - end_idx (int): 结束单词的索引。

    返回:
    - str: 给定索引范围内的内容。
    """

    # 按空格分割文本
    words = re.split(r'[ \u180e]+', text)

    # 提取指定范围内的内容
    extracted_content = " ".join(words[start_idx:end_idx])

    return extracted_content
if __name__ == '__main__':
    args = Args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.bert_dir)


    if args.do_train:
        raw_examples = Processor.get_examples(args.train_path, 'train')
        train_features = get_features(raw_examples, tokenizer, args)
        train_dataset = BertDataset(train_features)
        # labels = [example.seq_label for example in raw_examples]
        #
        # # 统计每个类别的数量
        # label_count = Counter(labels)
        #
        # # 计算每个样本的权重
        # total_samples = len(labels)
        # class_weights = {label: total_samples / count for label, count in label_count.items()}
        # sample_weights = np.array([class_weights[label] for label in labels])
        #
        # # 创建WeightedRandomSampler
        # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=total_samples, replacement=True)
        #
        # # 使用WeightedRandomSampler创建DataLoader
        # train_loader = DataLoader(train_dataset, batch_size=args.batchsize, sampler=sampler)
        train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)


    if args.do_eval:
        raw_examples = Processor.get_examples(args.test_path, 'test')
        test_features = get_features(raw_examples, tokenizer, args)
        test_dataset = BertDataset(test_features)
        test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=True)

    if args.do_test:
        raw_examples = Processor.get_examples(args.test_path, 'test')
        test_features = get_features(raw_examples, tokenizer, args)
        test_dataset = BertDataset(test_features)
        test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=True)


    model = BertForIntentClassificationAndSlotFilling(args)

    if args.load_model:
        model.load_state_dict(torch.load(args.load_dir))

    model.to(device)
    trainer = Trainer(model, args)

    if args.do_train:
        trainer.train(train_loader)

    if args.do_test:
        trainer.test(test_loader)

    if args.do_predict:
        # with open('./data/test.json','r',encoding='utf-8') as fp:
        #     pred_data = eval(fp.read())
        #     for i,p_data in enumerate(pred_data):
        #         text = p_data['text']
        #         print('=================================')
        #         print(text)
        #         trainer.predict(text)
        #         print('=================================')
        #         if i == 10:
        #             break
        while (1):
            txt =  input("输入语句：")
            # txt = "ᠮᠢᠨᠦ ᠳᠤᠷᠠᠲᠠᠢ ᠵᠢᠷᠤᠭᠲᠤ ᠳᠡᠪᠲᠡᠷ᠎ᠦ᠋ᠨ ᠸᠢᠳᠢᠣ᠎ᠶ᠋ᠢ ᠨᠡᠪᠲᠡᠷᠡᠭᠦᠯᠦᠶ᠎ᠡ"
            trainer.predict(txt)