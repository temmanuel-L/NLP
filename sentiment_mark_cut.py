import pandas as pd
import numpy as np
import sys
import math
import os
from smart_open import smart_open
import warnings
import re
import operator
import jieba
import ast
import time
import json
from algorithm import *


def run(input_data_name):
    # 该函数用来按照情感分值词表进行分词，并以情感分值词表作为正向过滤字典。
    # 目的在于为每篇评论生成一个新的特征，即所有词的打分之和
    # 预期上，负面的会大多为负分，正面大多为正分。中性在正、页之间徘徊，且绝对值较低。
    time_start1 = time.time()   # 第一起始时间
    Folder_Path_0 = os.path.abspath(os.path.join(os.getcwd(), ".."))
    os.chdir(Folder_Path_0)

    df_test = pd.read_excel(os.path.join(
        'data_input', input_data_name + '.xlsx'))

    list_content_mark = df_test.k_content.tolist()
    time_end1 = time.time()
    print('Step1: 加载源文件完毕，用时%.3f' % (time_end1 - time_start1))
    print('                                                         ')

    time_start2 = time.time()
    jieba.load_userdict(os.path.join('dict', 'userdict_sentiment_mark.txt'))
    stopwords = stop_words(os.path.join(
        'dict', 'stop_words.txt'))
    list_cut_mark1 = text_list_cut_jieba(stopwords, list_content_mark)
    time_end2 = time.time()
    print('Step2: 按照情感分值词典切词完毕，用时%.3f' % (time_end2 - time_start2))
    print('                                                         ')

    time_start3 = time.time()
    sentiment_mark_list = get_sentiment_mark_words_list()
    list_cut_mark2 = get_only_list_words_from_docs(
        list_cut_mark1, sentiment_mark_list)
    time_end3 = time.time()
    print('Step3: 使用情感分值词典进行正向过滤完毕，用时%.3f' % (time_end3 - time_start3))
    print('                                                         ')

    time_start4 = time.time()
    # 这一步是将list_cut_mark2中的词的分值相加，为此，需要先导入json的情感分值字典
    with open(os.path.join('dict', 'dict_sentiment_mark.json'), 'r') as load_f:
        sentiment_mark_dict = json.load(load_f)

    docs_sentiment_mark_list = sentiment_mark_count(
        sentiment_mark_dict, list_cut_mark2)

    docs_sentiment_mark_df = pd.Series(docs_sentiment_mark_list)
    docs_sentiment_mark_df.to_excel(os.path.join(
        'output', 'table', 'sentiment_mark_df.xlsx'))
    time_end4 = time.time()
    print('Step4: 各篇文章情感分值获取完毕，用时%.3f' % (time_end4 - time_start4))
    print('                                                         ')


if __name__ == "__main__":
    run('forum_0531_use')
