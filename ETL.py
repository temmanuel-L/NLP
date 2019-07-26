import pandas as pd
import numpy as np
import sys
import math
import os
import re
import ast
from ETL_algorithm import *
from gensim import corpora, models


def emotion_ETL(output_tryout_folder_name):
    Folder_Path_0 = os.path.abspath(os.path.join(os.getcwd(), ".."))
    os.chdir(Folder_Path_0)

    df_target_source = pd.read_excel(
        os.path.join('data_input', 'forum_0606.xlsx'))

    df_content_pos = pd.read_excel(os.path.join(
        'output_tryout', output_tryout_folder_name, 'table',
        'pos_content_matrix_docs_topics.xlsx'))

    df_content_neg = pd.read_excel(os.path.join(
        'output_tryout', output_tryout_folder_name, 'table',
        'neg_content_matrix_docs_topics.xlsx'))

    df_content_issue = pd.read_excel(os.path.join(
        'output_tryout', output_tryout_folder_name, 'table',
        'issue_content_matrix_docs_topics.xlsx'))

    df_title = pd.read_excel(os.path.join(
        'output_tryout', output_tryout_folder_name, 'table',
        'title_matrix_docs_topics.xlsx'))

    # 将 df_content, df_title的列名更改一下，以方便区别
    title_col_list_new = change_title_dataframe_col_name(df_title)
    df_title.columns = title_col_list_new
    del df_title['Title_dominant_topic']

    # 对df_content_pos, df_content_neg作同样处理
    pos_content_col_list_new = change_pos_content_dataframe_col_name(
        df_content_pos)
    df_content_pos.columns = pos_content_col_list_new
    del df_content_pos['Pos_Content_dominant_topic']

    neg_content_col_list_new = change_neg_content_dataframe_col_name(
        df_content_neg)
    df_content_neg.columns = neg_content_col_list_new
    del df_content_neg['Neg_Content_dominant_topic']

    issue_content_col_list_new = change_issue_content_dataframe_col_name(
        df_content_issue)
    df_content_issue.columns = issue_content_col_list_new
    del df_content_issue['Issue_Content_dominant_topic']

    # 将df_content, df_title拼接起来
    df_concat = pd.concat([df_title, df_content_pos, df_content_neg, df_content_issue], axis=1)

    # 将df_1的目标变量移植过来
    series_target = df_target_source['情感倾向']

    series_target_renew = series_target.apply(
        lambda x: emotion_words_transfer(x))

    series_target_renew.columns = ['target']

    # 经初步尝试，后面将series_target_renew通过concat方法不能直接与df_concat
    # 合并，可能是index的问题，为了快速解决，将series转成dataframe,通过merge合并
    df_target = pd.DataFrame(series_target_renew)

    df_target_reset = df_target.reset_index()   # 如此，right_on即为'index'

    # 对df_concat作同样处理
    df_concat_reset = df_concat.reset_index()
    df_concat_reset1 = df_concat_reset.reset_index()

    # 进行特征dataframe与目标变量dataframe的合并。
    df_dataset = pd.merge(df_concat_reset1, df_target_reset,
                          how='left', left_on='level_0', right_on='index')

    # 合并之后有三列是不需要的，即index_x, index_y, 'level_0'，删除
    df_dataset_1 = df_dataset
    col_drop_list = ['level_0', 'index_x', 'index_y']
    for col in col_drop_list:
        df_dataset_1.drop(col, inplace=True, axis=1)

    # 将目标变量“情感倾向”改一下名字
    df_dataset_1.rename(columns={'情感倾向': 'target'}, inplace=True)

    # 用ETL_algorithm中定义的函数创建目录
    check_and_mkdir(output_tryout_folder_name)

    # # 目前的想法是用帖子中出现的表情词语作为补充特征，为此需要先将output中的
    # # list_cut1_4和dictionary加载
    # list_cut1_4_temp = pd.read_csv(os.path.join(
    #     'output', 'list_cut', 'list_cut1_4.csv'), encoding='gbk', header=None)

    # # 从ast引入literal_eval，其作用等同于eval，但是更安全。
    # list_cut1_4_temp_1 = list_cut1_4_temp.iloc[:, 1].apply(
    #     lambda x: ast.literal_eval(x))
    # list_cut1_4 = list_cut1_4_temp_1.tolist()

    # # dictionary = corpora.Dictionary.load(
    # #     os.path.join('output', 'dictionary', 'dictionary1.txt'))

    # # 利用ETL_algorithm中的list_cut_freq_dict函数，将list_cut1_4中的
    # # 每篇文章改成去重词频字典
    # list_cut1_4_freq_dict = list_cut_freq_dict(list_cut1_4)

    # # print(list_cut1_4_freq_dict)

    # # 导入表情词特征清单wordcount_feature_list
    # expression_words_df = pd.read_excel(os.path.join(
    #     'dict', 'wordcount_feature_list.xlsx'))

    # # print(expression_words_df)

    # expression_words_list = expression_words_df.wordcount_name.tolist()

    # # 利用ETL_algorithm中的docs_expression_words_freq_matrix函数，
    # # 将lsit_cut1_4_freq_dict与word_count_list结合生成表情词特征矩阵
    # docs_expression_words_freq_matrix_list = docs_expression_words_freq_matrix(
    #     list_cut1_4_freq_dict, list_cut1_4, expression_words_list)

    # # print(docs_expression_words_freq_matrix_list)

    # # 将矩阵转成dataframe.
    # expression_words_feature = pd.DataFrame(
    #     docs_expression_words_freq_matrix_list)

    # # 将expression_words_feature列名改得有意义，并重新输出dataframe。
    # expression_words_feature_col_list = replace_dataframe_col_name(
    #     expression_words_feature)

    # expression_words_feature.columns = expression_words_feature_col_list

    # # print(expression_words_feature)

    # # 将df_dataset_1与expression_words_feature_new进行concat
    # df_dataset_0 = pd.concat(
    #     [expression_words_feature, df_dataset_1], axis=1)

    # 为了

    # 存储df_dataset_1，存储前将'target'列的数据类型改为int
    df_dataset_1['target'] = df_dataset_1['target'].astype("int")

    # 将table中生成的sentiment_mark_df作为一个特征与df_dataset_1合并
    df_sentiment_mark = pd.read_excel(
        os.path.join('output_tryout', output_tryout_folder_name,
                     'table', 'sentiment_mark_df.xlsx'))
    df_sentiment_mark.columns = ['doc_sentiment_mark']

    df_dataset_0 = pd.concat([df_sentiment_mark, df_dataset_1], axis=1)
    df_dataset_0.to_csv(os.path.join(
        'output_tryout', output_tryout_folder_name, 'train', 'dataset_concat.csv'))


if __name__ == "__main__":
    emotion_ETL('5th_try')
