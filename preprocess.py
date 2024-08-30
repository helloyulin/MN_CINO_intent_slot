import re
import torch
from transformers import BertTokenizer
from config import Args


class InputExample:
    def __init__(self, set_type, text, seq_label, token_label,domain_label):
        self.set_type = set_type
        self.text = text
        self.seq_label = seq_label
        self.token_label = token_label
        self.domain_label = domain_label

class InputFeature:
    def __init__(self,
                 input_ids,
                 attention_mask,
                 token_type_ids,
                 seq_label_ids,
                 token_label_ids,
                 domain_label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.seq_label_ids = seq_label_ids
        self.token_label_ids = token_label_ids
        self.domain_label_ids = domain_label_ids


class Processor:
    @classmethod
    def get_examples(cls, path, set_type):
        raw_examples = []
        with open(path, 'r',encoding='utf-8') as fp:
        # with open(path, 'r') as fp:
            data = eval(fp.read())
        for i, d in enumerate(data):
            text = d['text']
            seq_label = d['intent']
            domain_label = d['domain']
            token_label = d['slots']

            raw_examples.append(
                InputExample(
                    set_type,
                    text,
                    seq_label,
                    token_label,
                    domain_label
                )
            )
        return raw_examples

def calculate_positions(pattern, text):
    """
    Calculate the start and end positions of each matched entity in the text
    based on spaces.

    Parameters:
    - pattern (str): The regex pattern to match entities.
    - text (str): The input text to search for matches.

    Returns:
    - List of tuples containing (entity, start_position, end_position).
    """

    # Split the text by spaces to create a list of words and their positions
    words = text.split()
    word_positions = []

    current_position = 0
    for word in words:
        word_positions.append((word, current_position))
        current_position += len(word) + 1  # +1 for the space

    # Find all matches for the regex pattern
    matches = re.finditer(pattern, text)

    results = []
    for match in matches:
        entity = match.group()
        start = match.start()
        end = match.end()

        # Calculate the start and end positions based on word boundaries
        start_word_index = end_word_index = None

        for i, (word, position) in enumerate(word_positions):
            if position <= start < position + len(word):
                start_word_index = i
            if position <= end <= position + len(word):
                end_word_index = i + 1

        if start_word_index is not None and end_word_index is not None:
            results.append((entity, start_word_index, end_word_index))

    return results


def count_words(text):
    # 使用正则表达式按空格（包括 '\u180e'）分割文本
    words = re.split(r'[ \u180e]+', text)
    # 返回分割后的单词个数
    return len(words)
max = 0
def convert_example_to_feature(ex_idx, example, tokenizer, config):
    global max
    set_type = example.set_type
    text = example.text
    seq_label = example.seq_label
    token_label = example.token_label
    domain_label =example.domain_label
    seq_label_ids = config.seqlabel2id[seq_label]
    domain_label_ids = config.domainlabel2id[domain_label]
    token_label_ids = [0] * count_words(text)
    if len(text)>max:
        max=len(text)
        print("max___len:",max)
    for k, v in token_label.items():
        # print(k, v, text)
        matches = calculate_positions(v, text)
        for entity, start, end in matches:
            token_label_ids[start] = config.nerlabel2id['B-' + k]
            for i in range(start + 1, end):
                token_label_ids[i] = config.nerlabel2id['I-' + k]
    if len(token_label_ids) >= config.max_len - 2:
        token_label_ids = [0] + token_label_ids + [0]
    else:
        token_label_ids = [0] + token_label_ids + [0] + [0] * (config.max_len - len(token_label_ids) - 2)
    # print(token_label_ids)
    inputs1 = tokenizer.encode_plus(
        text=text,
        max_length=config.max_len,
        padding='max_length',
        truncation='only_first',
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    text = [i for i in text]
    # text = re.split(r'[ \u180e]+', text)
    # text = text.split(" ")
    inputs = tokenizer.encode_plus(
        text=text,
        max_length=config.max_len,
        padding='max_length',
        truncation='only_first',
        return_attention_mask=True,
        return_token_type_ids=True,
    )

    input_ids =  torch.tensor(inputs['input_ids'], requires_grad=False)
    attention_mask =  torch.tensor(inputs['attention_mask'], requires_grad=False)
    token_type_ids =  torch.tensor(inputs1['token_type_ids'], requires_grad=False)
    seq_label_ids  = torch.tensor(seq_label_ids, requires_grad=False)
    token_label_ids = torch.tensor(token_label_ids, requires_grad=False)
    domain_label_ids = torch.tensor(domain_label_ids, requires_grad=False)
    if ex_idx < 3:
        print(f'*** {set_type}_example-{ex_idx} ***')
        print(f'text: {text}')
        print(f'input_ids: {input_ids}')
        print(f'attention_mask: {attention_mask}')
        print(f'token_type_ids: {token_type_ids}')
        print(f'seq_label_ids: {seq_label_ids}')
        print(f'domain_label_ids: {domain_label_ids}')



    feature = InputFeature(
        input_ids,
        attention_mask,
        token_type_ids,
        seq_label_ids,
        token_label_ids,
        domain_label_ids
    )

    return feature


def get_features(raw_examples, tokenizer, args):
    features = []
    for i, example in enumerate(raw_examples):
        if len(example.text)<=155:
            feature = convert_example_to_feature(i, example, tokenizer, args)
            features.append(feature)
    return features


if __name__ == '__main__':
    args = Args()
    raw_examples = Processor.get_examples('data/test_process1.json', 'train')
    tokenizer = BertTokenizer.from_pretrained('../../model_hub/cino-bert-wwm-ext/')
    features = get_features(raw_examples, tokenizer, args)
