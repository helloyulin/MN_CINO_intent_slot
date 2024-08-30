import re


def split_sentence(sentence, delimiters):
    # 创建一个正则表达式模式，以捕获所有分隔符
    regex_pattern = '|'.join(map(re.escape, delimiters))
    words = re.split(f'({regex_pattern})', sentence)

    # 去除空字符串
    words = [word for word in words if word]
    return words


def join_sentence(words):
    return ''.join(words)


# 测试
sentence = "see you᠎again i am fine"
delimiters = [' ', '᠎']

words = split_sentence(sentence, delimiters)
print("拆分后的结果:", words)

reconstructed_sentence = join_sentence(words)
print("复原后的结果:", reconstructed_sentence)