import pandas as pd
import numpy as np
import sys
import math
import os
import re


def change_title_dataframe_col_name(df):
    col_list = df.columns.tolist()
    col_list_new = []
    for col in col_list:
        col = 'Title_' + col
        col_list_new.append(col)
    return col_list_new


def change_pos_content_dataframe_col_name(df):
    col_list = df.columns.tolist()
    col_list_new = []
    for col in col_list:
        col = 'Pos_Content_' + col
        col_list_new.append(col)
    return col_list_new


def change_neg_content_dataframe_col_name(df):
    col_list = df.columns.tolist()
    col_list_new = []
    for col in col_list:
        col = 'Neg_Content_' + col
        col_list_new.append(col)
    return col_list_new


def change_issue_content_dataframe_col_name(df):
    col_list = df.columns.tolist()
    col_list_new = []
    for col in col_list:
        col = 'Issue_Content_' + col
        col_list_new.append(col)
    return col_list_new


def regexp_find_results_for_list(regexp, doc):
    # 这个查询的函数与algorithm中的略有不同，后者查询的对象是分好的词，
    # 故返回全部信息，该查询面向的是句子，只返回需要的信息。
    result_list = []
    for x in doc:
        result = regexp.findall(x)
        if result:
            result_list.append(result)
    return result_list


def regexp_result_flatten(input_list):
    # flatten还是为了解决两层列表的问题
    output_list = []
    while True:
        if input_list == []:
            break
        for index, i in enumerate(input_list):
            if type(i) == list:
                input_list = i + input_list[index+1:]
                break
            else:
                output_list.append(i)
                input_list.pop(index)
                break
    return output_list


def drop_duplicates_and_delete_bracket(reg_result_1_1):
    # 将上面展成一维列表的词进行去重、删除括号操作
    # 先去重
    reg_result_1_new = []
    for exp in reg_result_1_1:
        if exp not in reg_result_1_new:
            reg_result_1_new.append(exp)

    # 再删除括号
    reg_result_1_new_1 = []
    for exp in reg_result_1_new:
        exp_1 = exp.replace('[', '').replace(']', '').replace(
            '【', '').replace('】', '')
        reg_result_1_new_1.append(exp_1)

    return reg_result_1_new_1


def emotion_words_transfer(x):
    # 定义一个函数，用于将情感倾向为正、负、中改成1、2、3数字
    x = x.replace('正', '1').replace('负', '0').replace('中', '2')
    return x


def reverse_emotion_words_transfer(x):
    # 定义一个函数，用于将情感倾向为正、负、中改成1、2、3数字
    x = x.replace('1', '正').replace('0', '负').replace('2', '中')
    return x


def check_and_mkdir(output_tryout_folder_name):
    # 将df_dataset_1存储
    mkdir_path = os.path.join(
        'output_tryout', output_tryout_folder_name, 'train')
    exist_judge = os.path.exists(mkdir_path)
    if not exist_judge:
        os.makedirs(mkdir_path)


def check_and_mkdir_predict(output_tryout_folder_name):
    mkdir_path = os.path.join(
        'output_tryout', output_tryout_folder_name, 'predict')
    exist_judge = os.path.exists(mkdir_path)
    if not exist_judge:
        os.makedirs(mkdir_path)


def list_cut_freq_dict(list_cut1_4):
    # 入参是最终的list_cut1_4
    # 该函数用于将双层列表list_cut1_4转换成为包含每篇文章词频字典的list
    text_list_with_freq = []
    for text in list_cut1_4:
        list_test_df = pd.Series(text)
        list_test_df_vc = pd.DataFrame(list_test_df.value_counts())
        list_test_df_vc_1 = list_test_df_vc.reset_index()
        list_test_df_vc_1_tuple = [tuple(x) for x in list_test_df_vc_1.values]
        dict_test = {}
        for x in list_test_df_vc_1_tuple:
            dict_test[x[0]] = x[1]
        text_list_with_freq.append(dict_test)

    return text_list_with_freq


def docs_expression_words_freq_matrix(list_cut1_4_freq_dict,
                                      list_cut1_4, expression_words_list):
    # 该函数用来将文章词频字典的列表，转换成文章与所有表情词的词频存在关系矩阵
    # 以text_list_with_freq(或者是真实的list_cut1_4_frq_dict)作为入参1，
    # 以list_cut1_4作为入参2，以情感词列表expression_words_list作为入参3.
    text_special_words_matrix = []
    for i in range(len(list_cut1_4_freq_dict)):
        text_special_words_relation = []
        for word in expression_words_list:
            if word in list_cut1_4[i]:
                text_special_words_relation.append(
                    list_cut1_4_freq_dict[i][word])
            else:
                text_special_words_relation.append(0)
        text_special_words_matrix.append(text_special_words_relation)

    return text_special_words_matrix


def replace_dataframe_col_name(expression_words_feature):
    # 入参为表情词生成的dataframe.
    expression_words_col_list = expression_words_feature.columns.tolist()

    expression_words_col_list_new = []
    for col_name in expression_words_col_list:
        col_name_new = 'exp_word_' + str(col_name)
        expression_words_col_list_new.append(col_name_new)

    return expression_words_col_list_new
