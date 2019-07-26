import pandas as pd
import numpy as np
import sys
import math
import os
import re
from ETL_algorithm import *

Folder_Path_0 = os.path.abspath(os.path.join(os.getcwd(), ".."))
os.chdir(Folder_Path_0)

df = pd.read_excel(os.path.join('data_input', 'wx_emotion_1.xlsx'))

# content的Series一定要进行str化，否则简短的评论如果只有数字后面会报错
df_content = df.content.astype("str")

content_list = df_content.tolist()

# 表情的词语基本都在[]内，但也有一些词语在【】内，当然这些可能是广告词
# 为此建立两个正则表达式
regexp_1 = re.compile(r'\[[\u4E00-\u9FA5]{1,}\]')
regexp_2 = re.compile(r'\【[\u4E00-\u9FA5]{1,}\】')

reg_result_1 = regexp_find_results_for_list(regexp_1, content_list)
reg_result_2 = regexp_find_results_for_list(regexp_2, content_list)

reg_result_1_1 = regexp_result_flatten(reg_result_1)
reg_result_2_1 = regexp_result_flatten(reg_result_2)

reg_result_1_new_1 = drop_duplicates_and_delete_bracket(reg_result_1_1)
reg_result_2_new_1 = drop_duplicates_and_delete_bracket(reg_result_2_1)

reg_result_final = reg_result_1_new_1+reg_result_2_new_1

reg_result_final_df = pd.DataFrame(reg_result_final)

reg_result_final_df.to_csv(os.path.join('dict', 'expression_words.csv'))
