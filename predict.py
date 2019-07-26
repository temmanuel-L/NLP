import pandas as pd
import numpy as np
import sys
import math
import os
from smart_open import smart_open
from gensim import corpora, models, similarities
from gensim.models import Phrases
from nltk.tokenize import RegexpTokenizer
import pkuseg
import matplotlib.pyplot as plt
import warnings
import re
import operator
import jieba
import ast
import time
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
import xgboost as xgb
from sklearn.externals import joblib
from algorithm import *
import ETL_algorithm

'''
该脚本为预测新增数据，因此，对新增数据的预处理，即：分词、词袋、语料库、TFIDF、主题、打分、ETL等所有
环节均需在此包括，唯一省略的是主题生成和监督学习的训练过程。
'''


def predict(input_data_name):
    # 该函数用于对新增未知数据进行预测。
    Folder_Path_0 = os.path.abspath(os.path.join(os.getcwd(), ".."))
    os.chdir(Folder_Path_0)

    print('--------------------Stage1: 导入数据集--------------------')
    time_start1 = time.time()

    # 先建一个predict文件夹
    check_and_mkdir_predict(['list_cut', 'corpus', 'table', 'result'])

    # 导入停用词
    stopwords = stop_words(os.path.join(
        'dict', 'stop_words.txt'))

    # 导入数据
    df_use = pd.read_excel(os.path.join(
        'data_input', input_data_name + '.xlsx'), encoding='utf-8')

    # docs1是取评论内容，是主题分类的主要部分
    docs1 = df_use.loc[:, 'k_content'].tolist()

    # docs2是每条评论的title，是主题分类的次要部分
    docs2 = df_use.loc[:, 'k_title'].tolist()

    # 加载包含正、负面相关词语的词典（约24000左右词语）。
    jieba.load_userdict(os.path.join(
        'dict', 'userdict_pos+neg+issue+mark.txt'))

    # 对docs1切词
    list_cut1_1 = text_list_cut_jieba(stopwords, docs1)
    # print('list_cut1_1 done!')

    # 对docs2切词
    list_cut2_1 = text_list_cut_jieba(stopwords, docs2)
    # print('list_cut2_1 done!')

    # 获得四类列表
    pos_list = get_positive_words_list()
    print('pos_list is built!')

    neg_list = get_negative_words_list()
    print('neg_list is built!')

    issue_list = get_issue_words_list()
    print('issue_list is built!')

    sentiment_mark_list = get_sentiment_mark_words_list()
    print('sentiment_mark_list is built!')

    # 用四类列表对list_cut1_1进行正向过滤
    list_cut1_1_pos = get_only_list_words_from_docs(list_cut1_1, pos_list)
    print('list_cut1_1_pos is built!')

    list_cut1_1_neg = get_only_list_words_from_docs(list_cut1_1, neg_list)
    print('list_cut1_1_neg is built!')

    list_cut1_1_issue = get_only_list_words_from_docs(list_cut1_1, issue_list)
    print('list_cut1_1_issue is built!')

    list_cut_mark2 = get_only_list_words_from_docs(
        list_cut1_1, sentiment_mark_list)

    # 将5个切词结果保存
    list_cut1_1_pos_series = pd.Series(list_cut1_1_pos)
    list_cut1_1_pos_series.to_csv(os.path.join(
        'predict', 'list_cut', 'list_cut1_1_pos.csv'))

    list_cut1_1_neg_series = pd.Series(list_cut1_1_neg)
    list_cut1_1_neg_series.to_csv(os.path.join(
        'predict', 'list_cut', 'list_cut1_1_neg.csv'))

    list_cut1_1_issue_series = pd.Series(list_cut1_1_issue)
    list_cut1_1_issue_series.to_csv(os.path.join(
        'predict', 'list_cut', 'list_cut1_1_issue.csv'))

    list_cut1_1_sentiment_mark_series = pd.Series(list_cut_mark2)
    list_cut1_1_sentiment_mark_series.to_csv(os.path.join(
        'predict', 'list_cut', 'list_cut1_1_sentiment_mark.csv'))

    list_cut2_1_series = pd.Series(list_cut2_1)
    list_cut2_1_series.to_csv(os.path.join(
        'predict', 'list_cut', 'list_cut2_1.csv'))

    time_end1 = time.time()
    print('Stage1完成，用时：%.3f' % (time_end1-time_start1))
    print('                                                                       ')

    print('-----------------Stage2-1: 过滤模块——正则前处理--------------------------')
    time_start2_1 = time.time()

    # 利用alogrithm中的word_combination函数，将切记工具切开的，
    # 但经常一起出现的词用'_'连接起来。
    list_cut1_1_pos_new = add_word_combination(list_cut1_1_pos, 2)
    list_cut1_1_neg_new = add_word_combination(list_cut1_1_neg, 2)
    list_cut1_1_issue_new = add_word_combination(list_cut1_1_issue, 2)
    list_cut2_1_new = add_word_combination(list_cut2_1, 2)

    # ------------------------------------------------------------------------------
    #  由于目前每篇评论切记的结果，均已直接取自于两个白名单字典，故后面的很多步骤可略去  -
    #  但是list_cut2_4的流程可以保留下来，主要是为了方便。                           -
    # ------------------------------------------------------------------------------
    # # 应用algorithm中的去数字函数，得到更新的切词结果。
    # list_cut1_2 = drop_number(list_cut1_1_new)
    list_cut2_2 = drop_number(list_cut2_1_new)

    time_end2_1 = time.time()
    print('Stage2-1完成，用时：%.3f' % (time_end2_1-time_start2_1))
    print('                                                                       ')

    print('-----------------Stage2-2: 过滤模块——正则表达式筛选-----------------------')
    time_start2_2 = time.time()
    # 利用algorithm的special_regexp_list()获得特殊正则规则的列表
    regexp_special_list = special_regexp_list()

    screened_words = stop_words(os.path.join(
        'dict', 'screened_words_emotion.txt'))

    # 由于screened_words不是列表形式，需要进行split处理。
    screened_words_new = screened_words.split('\n')

    # 正则表达式中，汉字需表式成UTF码，为此导入汉字与UTF码的对应表
    char_utf_df = pd.read_excel(os.path.join('dict', 'char_utf_table.xlsx'))

    # 利用algorithm中的char_utf_df_processing函数，得到指定汉字转化为UTF码的列表
    utf_screen_list = char_utf_df_processing(char_utf_df, screened_words_new)

    # 利用algorithm中的special_regexp_list函数，得到找到包含指定汉字的词语
    regexp_general_list = general_regexp_list(utf_screen_list)

    # 将将regexp_general_list与regexp_special_lsit合并
    regexp_list_concat = regexp_special_list + regexp_general_list

    # # 分别在list_cut1_2与list_cut2_2中进行regexp_list_concat包含词的去除操作
    # # 需调用algorithm中的list_cut_screened_regexp函数
    # list_cut1_3 = list_cut_screened_regexp(list_cut1_2, regexp_list_concat)
    list_cut2_3 = list_cut_screened_regexp(list_cut2_2, regexp_list_concat)

    time_end2_2 = time.time()
    print('Stage2-2完成，用时：%.3f' % (time_end2_2-time_start2_2))
    print('                                                                       ')

    print('-----------------Stage2-3: 过滤模块——开源字典筛选------------------------')
    time_start2_3 = time.time()

    # 与聆听项目不同，长度为1的词显然不能去除
    # 同时，显然也不适合像聆听项目一样对black_list词语作筛选去除，因为其中含有大量程度副词和情感类词语
    # 地名、姓式、hf_useless_words和dealer还是可以去除的。与此对应的。algorithm中的函数需做相应调整。
    address_to_be_screened_list, firstname_to_be_screened_list, hf_normal_to_be_screened_list, dealer_to_be_screened_list = other_to_be_screened_list(
        os.path.join('dict', 'name_of_place.xlsx'),
        os.path.join('dict', 'Chinese_Family_Name（1k）.xlsx'),
        os.path.join('dict', 'hf_useless_words.xlsx'),
        os.path.join('dict', 'dealer.xlsx'))

    open_dict_to_be_screened_list = [address_to_be_screened_list,
                                     firstname_to_be_screened_list,
                                     hf_normal_to_be_screened_list,
                                     dealer_to_be_screened_list]

    # # 分别对list_cut1_3， list_cut2_3的词语进行以上四个字典的过滤。
    # list_cut1_4 = list_cut_screened_dict(
    #     list_cut1_3, open_dict_to_be_screened_list)

    list_cut2_4 = list_cut_screened_dict(
        list_cut2_3, open_dict_to_be_screened_list)

    # 保存5个切词列表
    list_cut1_1_pos_new_series = pd.Series(list_cut1_1_pos_new)
    list_cut1_1_pos_new_series.to_csv(os.path.join(
        'predict', 'list_cut', 'list_cut1_1_pos_new.csv'))

    # list_cut1_1_pos_new_temp = pd.read_csv(os.path.join(
    #     'predict', 'list_cut', 'list_cut1_1_pos_new.csv'), encoding='gbk', header=None)
    # list_cut1_1_pos_new_temp_1 = list_cut1_1_pos_new_temp.iloc[:, 1].apply(
    #     lambda x: ast.literal_eval(x))
    # list_cut1_1_pos_new = list_cut1_1_pos_new_temp_1.tolist()

    list_cut1_1_neg_new_series = pd.Series(list_cut1_1_neg_new)
    list_cut1_1_neg_new_series.to_csv(os.path.join(
        'predict', 'list_cut', 'list_cut1_1_neg_new.csv'))

    # list_cut1_1_neg_new_temp = pd.read_csv(os.path.join(
    #     'predict', 'list_cut', 'list_cut1_1_neg_new.csv'), encoding='gbk', header=None)
    # list_cut1_1_neg_new_temp_1 = list_cut1_1_neg_new_temp.iloc[:, 1].apply(
    #     lambda x: ast.literal_eval(x))
    # list_cut1_1_neg_new = list_cut1_1_neg_new_temp_1.tolist()

    list_cut2_4_series = pd.Series(list_cut2_4)
    list_cut2_4_series.to_csv(os.path.join(
        'predict', 'list_cut', 'list_cut2_4.csv'))

    # list_cut2_4_temp = pd.read_csv(os.path.join(
    #     'predict', 'list_cut', 'list_cut2_4.csv'), encoding='gbk', header=None)
    # list_cut2_4_temp_1 = list_cut2_4_temp.iloc[:, 1].apply(
    #     lambda x: ast.literal_eval(x))
    # list_cut2_4 = list_cut2_4_temp_1.tolist()
    list_cut1_1_issue_new_series = pd.Series(list_cut1_1_issue_new)
    list_cut1_1_issue_new_series.to_csv(os.path.join(
        'predict', 'list_cut', 'list_cut1_1_issue_new.csv'))

    with open(os.path.join('dict', 'dict_sentiment_mark.json'), 'r') as load_f:
        sentiment_mark_dict = json.load(load_f)

    docs_sentiment_mark_list = sentiment_mark_count(
        sentiment_mark_dict, list_cut_mark2)

    docs_sentiment_mark_df = pd.Series(docs_sentiment_mark_list)
    docs_sentiment_mark_df.to_excel(os.path.join(
        'predict', 'table', 'sentiment_mark_df.xlsx'))

    time_end2_3 = time.time()
    print('Stage2-3完成，用时：%.3f' % (time_end2_3-time_start2_3))
    print('                                                                ')

    print('-----------------------Stage3: 词袋模块-------------------------')
    time_start3 = time.time()

    # 与theme.py的词袋模块不同的是，这里需要直接使用theme.py生成的dictionary
    # 及tfidf模型，而corpus是以原来的dictionary为基础，在新语料的基础上构建的
    # 用resource_settled中的词袋，生成新的语料库
    # 首先是pos的corpus。--------------------------------------------------
    dictionary1_pos = corpora.Dictionary.load(os.path.join(
        'resource_settled', 'dictionary1_pos.txt'))

    corpus1_pos = corpora.MmCorpus(os.path.join(
        'resource_settled', 'corpus1_pos.mm'))

    corpus1_pos_new = build_corpus_for_new_docs(
        dictionary1_pos, list_cut1_1_pos_new)

    corpus_tfidf1_pos_new = build_corpus_tfidf_for_new_docs(
        corpus1_pos, corpus1_pos_new)

    corpora.MmCorpus.serialize(os.path.join(
        'predict', 'corpus', 'corpus1_pos_new.mm'), corpus1_pos_new)    # 保存
    corpora.MmCorpus.serialize(os.path.join(
        'predict', 'corpus', 'corpus_tfidf1_pos_new.mm'), corpus_tfidf1_pos_new)    # 保存

    # 然后是neg的corpus----------------------------------------------------
    dictionary1_neg = corpora.Dictionary.load(os.path.join(
        'resource_settled', 'dictionary1_neg.txt'))

    corpus1_neg = corpora.MmCorpus(os.path.join(
        'resource_settled', 'corpus1_neg.mm'))

    corpus1_neg_new = build_corpus_for_new_docs(
        dictionary1_neg, list_cut1_1_neg_new)

    corpus_tfidf1_neg_new = build_corpus_tfidf_for_new_docs(
        corpus1_neg, corpus1_neg_new)

    corpora.MmCorpus.serialize(os.path.join(
        'predict', 'corpus', 'corpus1_neg_new.mm'), corpus1_neg_new)    # 保存
    corpora.MmCorpus.serialize(os.path.join(
        'predict', 'corpus', 'corpus_tfidf1_neg_new.mm'), corpus_tfidf1_neg_new)    # 保存

    # issue的corpus----------------------------------------------------
    dictionary1_issue = corpora.Dictionary.load(os.path.join(
        'resource_settled', 'dictionary1_issue.txt'))

    corpus1_issue = corpora.MmCorpus(os.path.join(
        'resource_settled', 'corpus1_issue.mm'))

    corpus1_issue_new = build_corpus_for_new_docs(
        dictionary1_issue, list_cut1_1_issue_new)

    corpus_tfidf1_issue_new = build_corpus_tfidf_for_new_docs(
        corpus1_issue, corpus1_issue_new)

    corpora.MmCorpus.serialize(os.path.join(
        'predict', 'corpus', 'corpus1_issue_new.mm'), corpus1_issue_new)    # 保存
    corpora.MmCorpus.serialize(os.path.join(
        'predict', 'corpus', 'corpus_tfidf1_issue_new.mm'), corpus_tfidf1_issue_new)    # 保存

    # title的corpus----------------------------------------------------
    dictionary2 = corpora.Dictionary.load(os.path.join(
        'resource_settled', 'dictionary2.txt'))

    corpus2 = corpora.MmCorpus(os.path.join(
        'resource_settled', 'corpus2.mm'))

    corpus2_new = build_corpus_for_new_docs(dictionary2, list_cut2_4)

    corpus_tfidf2_new = build_corpus_tfidf_for_new_docs(corpus2, corpus2_new)

    corpora.MmCorpus.serialize(os.path.join(
        'predict', 'corpus', 'corpus2_new.mm'), corpus2_new)    # 保存
    corpora.MmCorpus.serialize(os.path.join(
        'predict', 'corpus', 'corpus_tfidf2_new.mm'), corpus_tfidf2_new)    # 保存

    time_end3 = time.time()
    print('Stage3完成，用时：%.3f' % (time_end3-time_start3))
    print('                                        ')

    print('---------------------Stage4: 导入预训练主题模型------------------------')
    time_start4 = time.time()

    # 通过algorithm的get_the_only_num_of_topcis得到合适的num_topics值
    best_model_num_topics1_pos = 25

    # 打印一下最佳主题数量
    print('For content_pos, The best num_topics is: %d.' %
          best_model_num_topics1_pos)

    # 从存储的全部model中，取出最佳model
    model1_pos = models.ldamodel.LdaModel.load(os.path.join(
        'resource_settled', 'pos_num_topic' + str(best_model_num_topics1_pos) + '.model'))

    # 消极评论内容(content_neg)----------------------------------------------------------------------------
    best_model_num_topics1_neg = 10

    # # 打印一下最佳主题数量
    print('For content_neg, The best num_topics is: %d.' %
          best_model_num_topics1_neg)

    # 从存储的全部model中，取出最佳model
    model1_neg = models.ldamodel.LdaModel.load(os.path.join(
        'resource_settled', 'neg_num_topic' + str(best_model_num_topics1_neg) + '.model'))

    # 问题内容（issue)----------------------------------------------------------------------------
    # 通过algorithm的get_the_only_num_of_topcis得到合适的num_topics值
    best_model_num_topics1_issue = 25

    # # 打印一下最佳主题数量
    print('For content_issue, The best num_topics is: %d.' %
          best_model_num_topics1_issue)

    # 从存储的全部model中，取出最佳model
    model1_issue = models.ldamodel.LdaModel.load(os.path.join(
        'resource_settled', 'issue_num_topic' + str(best_model_num_topics1_issue) + '.model'))

    # 标题内容(title)----------------------------------------------------------------------------
    # 通过algorithm的get_the_only_num_of_topcis得到合适的num_topics值
    best_model_num_topics2 = 8

    # 打印一下最佳主题数量
    print('For title, The best num_topics is: %d.' % best_model_num_topics2)

    # 从存储的全部model中，取出最佳model
    model2 = models.ldamodel.LdaModel.load(os.path.join(
        'resource_settled', 'title_num_topic' + str(best_model_num_topics2) + '.model'))

    time_end4 = time.time()
    print('Stage4完成，用时：%.3f' % (time_end4-time_start4))
    print('                                                            ')

    print('---------------------Stage5: 生成主题模型特征------------------------')
    time_start5 = time.time()
    # 至此，构建matrix_table的所有入参，即model, corpus_tfidf_new, docs均已齐备，
    # 可以分别生成matrix_table
    doc_topic_ratio_df1_pos = matrix_table(
        model1_pos, corpus_tfidf1_pos_new, docs1)
    doc_topic_ratio_df1_pos.to_excel(os.path.join(
        'predict', 'table', 'pos_content_matrix_docs_topics.xlsx'))     # 保存

    # neg的matrix_table
    doc_topic_ratio_df1_neg = matrix_table(
        model1_neg, corpus_tfidf1_neg_new, docs1)
    doc_topic_ratio_df1_neg.to_excel(os.path.join(
        'predict', 'table', 'neg_content_matrix_docs_topics.xlsx'))

    # issue的matrix_table
    doc_topic_ratio_df1_issue = matrix_table(
        model1_issue, corpus_tfidf1_issue_new, docs1)
    doc_topic_ratio_df1_issue.to_excel(os.path.join(
        'predict', 'table', 'issue_content_matrix_docs_topics.xlsx'))

    # title的matrix_table
    doc_topic_ratio_df2 = matrix_table(model2, corpus_tfidf2_new, docs2)
    doc_topic_ratio_df2.to_excel(os.path.join(
        'predict', 'table', 'title_matrix_docs_topics.xlsx'))

    time_end5 = time.time()
    print('Stage5完成，用时：%.3f' % (time_end5-time_start5))
    print('                                                            ')

    print('---------------------Stage6: 特征整合------------------------')
    time_start6 = time.time()

    # 为了延续之前代码变量名，将上面的doc_topic_ratio_df1_pos， ……进行变量替换。
    df_content_pos = pd.read_excel(os.path.join('predict', 'table',
                                                'pos_content_matrix_docs_topics.xlsx'))

    df_content_neg = pd.read_excel(os.path.join('predict', 'table',
                                                'neg_content_matrix_docs_topics.xlsx'))

    df_content_issue = pd.read_excel(os.path.join('predict', 'table',
                                                  'issue_content_matrix_docs_topics.xlsx'))

    df_title = pd.read_excel(os.path.join('predict', 'table',
                                          'title_matrix_docs_topics.xlsx'))

    # 将 df_content, df_title的列名更改一下，以方便区别
    title_col_list_new = ETL_algorithm.change_title_dataframe_col_name(
        df_title)
    df_title.columns = title_col_list_new
    del df_title['Title_dominant_topic']

    # 对df_content_pos, df_content_neg作同样处理
    pos_content_col_list_new = ETL_algorithm.change_pos_content_dataframe_col_name(
        df_content_pos)
    df_content_pos.columns = pos_content_col_list_new
    del df_content_pos['Pos_Content_dominant_topic']

    neg_content_col_list_new = ETL_algorithm.change_neg_content_dataframe_col_name(
        df_content_neg)
    df_content_neg.columns = neg_content_col_list_new
    del df_content_neg['Neg_Content_dominant_topic']

    issue_content_col_list_new = ETL_algorithm.change_issue_content_dataframe_col_name(
        df_content_issue)
    df_content_issue.columns = issue_content_col_list_new
    del df_content_issue['Issue_Content_dominant_topic']

    # 将df_content, df_title拼接起来
    df_concat = pd.concat(
        [df_title, df_content_pos, df_content_neg, df_content_issue], axis=1)

    # # 将df_1的目标变量移植过来
    # series_target = df_use['情感倾向']

    # series_target_renew = series_target.apply(
    #     lambda x: ETL_algorithm.emotion_words_transfer(x))

    # series_target_renew.columns = ['target']

    # # 经初步尝试，后面将series_target_renew通过concat方法不能直接与df_concat
    # # 合并，可能是index的问题，为了快速解决，将series转成dataframe,通过merge合并
    # df_target = pd.DataFrame(series_target_renew)

    # df_target_reset = df_target.reset_index()   # 如此，right_on即为'index'

    # # 对df_concat作同样处理
    # df_concat_reset = df_concat.reset_index()
    # df_concat_reset1 = df_concat_reset.reset_index()

    # # 进行特征dataframe与目标变量dataframe的合并。
    # df_dataset = pd.merge(df_concat_reset1, df_target_reset,
    #                       how='left', left_on='level_0', right_on='index')

    # # 合并之后有三列是不需要的，即index_x, index_y, 'level_0'，删除
    # df_dataset_1 = df_dataset
    # col_drop_list = ['level_0', 'index_x', 'index_y']
    # for col in col_drop_list:
    #     df_dataset_1.drop(col, inplace=True, axis=1)

    # # 将目标变量“情感倾向”改一下名字
    # df_dataset_1.rename(columns={'情感倾向': 'target'}, inplace=True)

    # # 目标变量列设置为int类型。
    # df_dataset_1['target'] = df_dataset_1['target'].astype("int")

    # 将table中生成的sentiment_mark_df作为一个特征与df_dataset_1合并
    df_sentiment_mark = pd.read_excel(
        os.path.join('predict', 'table', 'sentiment_mark_df.xlsx'))

    # 为了将df_sentiment_mark与df_concat的index一致，需进行调整
    df_sentiment_mark_list = df_sentiment_mark.iloc[:, 0].tolist()
    df_sentiment_mark_index = [
        "Doc" + str(i) for i in range(df_sentiment_mark.shape[0])]
    df_sentiment_mark_new = pd.DataFrame(
        df_sentiment_mark_list, index=df_sentiment_mark_index, columns=['doc_sentiment_mark'])

    df_dataset_0 = pd.concat([df_sentiment_mark_new, df_concat], axis=1)
    df_dataset_0.to_csv(os.path.join('predict', 'table', 'dataset_concat.csv'))

    time_end6 = time.time()
    print('Stage6完成，用时：%.3f' % (time_end6-time_start6))
    print('                                                            ')

    print('---------------------Stage7: 新文本情感预测------------------------')
    time_start7 = time.time()
    doc_emotion_set = pd.read_csv(os.path.join(
        'predict', 'table', 'dataset_concat.csv'))

    model_emotion = joblib.load(os.path.join(
        'resource_settled', 'model_emotion.model'))

    all_x_test = doc_emotion_set.iloc[:, 1:]
    all_y_pred = pd.Series(model_emotion.predict(all_x_test))
    all_y_pred_df = pd.DataFrame(all_y_pred)
    all_y_pred_df.columns = ['emotion_prediction']
    all_y_pred_df['emotion_prediction'] = all_y_pred_df['emotion_prediction'].astype("str")
    all_y_pred_df['emotion_prediction'] = all_y_pred_df['emotion_prediction'].apply(
        lambda x: ETL_algorithm.reverse_emotion_words_transfer(x))
    origin_df = pd.read_excel(os.path.join(
        'data_input', input_data_name + '.xlsx'))
    origin_add_y_pred = pd.concat([origin_df, all_y_pred_df], axis=1)

    origin_add_y_pred.to_excel(os.path.join(
        'predict', 'result', 'origin_add_y_pred.xlsx'))

    time_end7 = time.time()
    print('Stage7完成，用时：%.3f' % (time_end7-time_start7))
    print('                                                            ')


if __name__ == "__main__":
    predict('forum_0606')
