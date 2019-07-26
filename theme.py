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
from algorithm import *
from var import *

# 初步确定情感模型完全基于主题模型划分的主题，因此该部分暂时不要考虑特征的问题
# 应专注于将主题划好。由于不是循环对很多问题进行主题建模，因此微信文章的划分可以
# 考虑采用传统的人工方式。
# 对于微信的数据字段的选择，可以考虑使用两个字段，字段1：评论内容，字段2：title
# 不只对于微信数据，对于论坛数据，也是有title的。
# theme.py属于奥迪聆听项目主题分类的简化版本。
# 相比于前面的情感的theme.py文件，该版本使用了白名单反向过滤方法，这样可以保留更多的信息
# 防止正向过滤法筛选后，较多评论无词的结果。


def run(input_data_name):
    # 选取文件名作为入参，以方便快速适应不同的文件名
    # 根目录地址，问题序号，全文档词语tfidf值
    print('-----------------Stage1: 切词模块--------------------------')
    time_start1 = time.time()

    # 定义待获取数据的地址，为执行脚本上级目录的data_by_issue文件夹
    Folder_Path_0 = os.path.abspath(os.path.join(os.getcwd(), ".."))
    os.chdir(Folder_Path_0)

    # 确定output文件夹输出的内容，初步认为有corpus, dictionary, list_cut, model, table五个文件夹
    # 由于后面将对评论内容（content字段），帖子标题（title字段）分别进行数据读取、
    # 切词、建词袋、建语料库、建模等操作，而model存储起来，数量巨大
    # 故分别建model_content, model_title两个子文件夹。
    check_and_mkdir(['list_cut', 'dictionary', 'corpus', 'model_content_pos', 'model_content_neg', 'model_content_issue',
                     'model_title', 'table', 'pic_content_pos', 'pic_content_neg', 'pic_content_issue', 'pic_title'])

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
    jieba.load_userdict(os.path.join('dict', 'userdict_pos+neg+issue.txt'))

    # 对docs1切词
    list_cut1_1 = text_list_cut_jieba(stopwords, docs1)
    # print('list_cut1_1 done!')

    # 对docs2切词
    list_cut2_1 = text_list_cut_jieba(stopwords, docs2)
    # print('list_cut2_1 done!')

    # 对于list_cut1_1，我们希望分别进行两次独立的处理，一次为了只取出正面的情感词，主题聚类后得一种特征；
    # 另一次为了只取出负面的情感词，主题聚类后得另一种特征。当然，对于title的聚类方式，与原来一致。
    # 为此，先得到pos_List, neg_list
    pos_list = get_positive_words_list()
    print('pos_list is built!')

    neg_list = get_negative_words_list()
    print('neg_list is built!')

    issue_list = get_issue_words_list()
    print('issue_list is built!')

    list_cut1_1_pos = get_only_list_words_from_docs(list_cut1_1, pos_list)
    print('list_cut1_1_pos is built!')

    list_cut1_1_neg = get_only_list_words_from_docs(list_cut1_1, neg_list)
    print('list_cut1_1_neg is built!')

    list_cut1_1_issue = get_only_list_words_from_docs(list_cut1_1, issue_list)
    print('list_cut1_1_issue is built!')

    # 将三个list进行保存
    list_cut1_1_pos_series = pd.Series(list_cut1_1_pos)
    list_cut1_1_pos_series.to_csv(os.path.join(
        'output', 'list_cut', 'list_cut1_1_pos.csv'))

    list_cut1_1_neg_series = pd.Series(list_cut1_1_neg)
    list_cut1_1_neg_series.to_csv(os.path.join(
        'output', 'list_cut', 'list_cut1_1_neg.csv'))

    list_cut1_1_issue_series = pd.Series(list_cut1_1_issue)
    list_cut1_1_issue_series.to_csv(os.path.join(
        'output', 'list_cut', 'list_cut1_1_issue.csv'))

    list_cut2_1_series = pd.Series(list_cut2_1)
    list_cut2_1_series.to_csv(os.path.join(
        'output', 'list_cut', 'list_cut2_1.csv'))

    time_end1 = time.time()
    print('Stage1完成，用时：%.3f' % (time_end1-time_start1))
    print('                                                                       ')

    # 接下来，考虑一个问题，我们是否需要对切词结果进行正则化，答案是肯定需要的，但是相比于奥迪聆听项目，
    # 正则的规则和数量也必须有一些不同。
    # 1. 对于日期、时间类的切词结果，该去除还是去除
    #
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
    # 对于list_cut1_1_pos_new和list_cut1_1_neg_new,正则模块本次不使用，但了保证顺利得到
    # list_cut2_4，正则、开源字典等模块还是需要保留。

    # 利用algorithm的special_regexp_list()获得特殊正则规则的列表
    regexp_special_list = special_regexp_list()

    # 接着定义去除指定汉字的表达式，为此需要先引入包含这些汉字的外部文件
    # 情感分析的screen_words，与奥迪聆听项目还是不同，所以其screen_words，需要增、删一些东西。
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

    # 开源字典也无需过滤了。
    # 与聆听项目不同，长度为1的词显然不能去除
    # 同时，显然也不适合像聆听项目一样对black_list词语作筛选去除，因为其中含有大量程度副词和情感类词语
    # 但是，地名、姓式、hf_useless_words和dealer还是可以去除的。与此对应的。algorithm中的函数需做相应调整。
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

    # 奥迪聆听项目不同，截至目前的list_cut1_4与list_cut2_4已经是OK的分词结果了
    # 聆听项目由于要进行未知、关键的、非普通的信息挖掘，因此需要进行大量无关词过滤
    # 并将业务关心的词排在前面 ，故其list_cut2还需要通过转为corpus之后，以corpus_tfidf
    # 的权重取每篇文章的前20位，并以其重新调整list_cut2并生成list_cut2_new
    # 结论，list_cut1_4, list_cut2_4在分词层面已经结束。
    # 对于pos和neg的最终list_cut，也需要保存
    list_cut1_1_pos_new_series = pd.Series(list_cut1_1_pos_new)
    list_cut1_1_pos_new_series.to_csv(os.path.join(
        'output', 'list_cut', 'list_cut1_1_pos_new.csv'))

    # list_cut1_1_pos_new_temp = pd.read_csv(os.path.join(
    #     'output', 'list_cut', 'list_cut1_1_pos_new.csv'), encoding='gbk', header=None)
    # list_cut1_1_pos_new_temp_1 = list_cut1_1_pos_new_temp.iloc[:, 1].apply(
    #     lambda x: ast.literal_eval(x))
    # list_cut1_1_pos_new = list_cut1_1_pos_new_temp_1.tolist()

    list_cut1_1_neg_new_series = pd.Series(list_cut1_1_neg_new)
    list_cut1_1_neg_new_series.to_csv(os.path.join(
        'output', 'list_cut', 'list_cut1_1_neg_new.csv'))

    # list_cut1_1_neg_new_temp = pd.read_csv(os.path.join(
    #     'output', 'list_cut', 'list_cut1_1_neg_new.csv'), encoding='gbk', header=None)
    # list_cut1_1_neg_new_temp_1 = list_cut1_1_neg_new_temp.iloc[:, 1].apply(
    #     lambda x: ast.literal_eval(x))
    # list_cut1_1_neg_new = list_cut1_1_neg_new_temp_1.tolist()

    list_cut2_4_series = pd.Series(list_cut2_4)
    list_cut2_4_series.to_csv(os.path.join(
        'output', 'list_cut', 'list_cut2_4.csv'))

    # list_cut2_4_temp = pd.read_csv(os.path.join(
    #     'output', 'list_cut', 'list_cut2_4.csv'), encoding='gbk', header=None)
    # list_cut2_4_temp_1 = list_cut2_4_temp.iloc[:, 1].apply(
    #     lambda x: ast.literal_eval(x))
    # list_cut2_4 = list_cut2_4_temp_1.tolist()
    list_cut1_1_issue_new_series = pd.Series(list_cut1_1_issue_new)
    list_cut1_1_issue_new_series.to_csv(os.path.join(
        'output', 'list_cut', 'list_cut1_1_issue_new.csv'))

    time_end2_3 = time.time()
    print('Stage2-3完成，用时：%.3f' % (time_end2_3-time_start2_3))
    print('                                                                ')

    print('-----------------------Stage3: 词袋模块-------------------------')
    time_start3 = time.time()

    # 以list_cut1_4, list_cut2_4为基础，建立字典。
    # 因为title的精度会高一些，content的精度会低，
    # 所以基于content和title分出的类别不能混淆，
    # 故应该将content的docs与title的docs完全分开，无论是从字典还是从语料库的角度

    dictionary1_pos = corpora.Dictionary(list_cut1_1_pos_new)
    dictionary1_neg = corpora.Dictionary(list_cut1_1_neg_new)
    dictionary1_issue = corpora.Dictionary(list_cut1_1_issue_new)
    dictionary2 = corpora.Dictionary(list_cut2_4)

    dictionary1_pos.save(os.path.join('output', 'dictionary',
                                  'dictionary1_pos.txt'))  # 保存生成的词典
    dictionary1_neg.save(os.path.join('output', 'dictionary',
                                  'dictionary1_neg.txt'))
    dictionary1_issue.save(os.path.join('output', 'dictionary',
                                  'dictionary1_issue.txt'))
    dictionary2.save(os.path.join('output', 'dictionary',
                                  'dictionary2.txt'))

    # dictionary = corpora.Dictionary.load(os.path.join(
    #     'output', 'dictionary', 'dictionary1.txt'))  # 加载

    corpus1_pos = [dictionary1_pos.doc2bow(text) for text in list_cut1_1_pos_new]
    corpora.MmCorpus.serialize(os.path.join(
        'output', 'corpus', 'corpus1_pos.mm'), corpus1_pos)  # 保存生成的语料
    # corpus1 = corpora.MmCorpus(os.path.join(
    #     'output', 'corpus', 'corpus1.mm'))  # 加载一般性的corpus

    corpus1_neg = [dictionary1_neg.doc2bow(text) for text in list_cut1_1_neg_new]
    corpora.MmCorpus.serialize(os.path.join(
        'output', 'corpus', 'corpus1_neg.mm'), corpus1_neg)

    corpus1_issue = [dictionary1_issue.doc2bow(text) for text in list_cut1_1_issue_new]
    corpora.MmCorpus.serialize(os.path.join(
        'output', 'corpus', 'corpus1_issue.mm'), corpus1_issue)

    corpus2 = [dictionary2.doc2bow(text) for text in list_cut2_4]
    corpora.MmCorpus.serialize(os.path.join(
        'output', 'corpus', 'corpus2.mm'), corpus2)  # 保存生成的语料
    # corpus2 = corpora.MmCorpus(os.path.join(
    #     'output', 'corpus', 'corpus2.mm'))  # 加载一般性的corpus

    # 加载tfidf
    tfidf1_pos = models.TfidfModel(corpus1_pos)
    tfidf1_neg = models.TfidfModel(corpus1_neg)
    tfidf1_issue = models.TfidfModel(corpus1_issue)
    tfidf2 = models.TfidfModel(corpus2)

    corpus_tfidf1_pos = tfidf1_pos[corpus1_pos]
    corpora.MmCorpus.serialize(os.path.join(
        'output', 'corpus', 'corpus_tfidf1_pos.mm'), corpus_tfidf1_pos)  # 将计算权重的corpus_tfidf再保存
    # corpus_tfidf1 = corpora.MmCorpus(os.path.join(
    #     'output', 'corpus', 'corpus_tfidf1.mm'))  # 加载corpus_tfidf

    corpus_tfidf1_neg = tfidf1_neg[corpus1_neg]
    corpora.MmCorpus.serialize(os.path.join(
        'output', 'corpus', 'corpus_tfidf1_neg.mm'), corpus_tfidf1_neg)

    corpus_tfidf1_issue = tfidf1_pos[corpus1_issue]
    corpora.MmCorpus.serialize(os.path.join(
        'output', 'corpus', 'corpus_tfidf1_issue.mm'), corpus_tfidf1_issue)

    corpus_tfidf2 = tfidf2[corpus2]
    corpora.MmCorpus.serialize(os.path.join(
        'output', 'corpus', 'corpus_tfidf2.mm'), corpus_tfidf2)  # 将计算权重的corpus_tfidf再保存
    # corpus_tfidf1 = corpora.MmCorpus(os.path.join(
    #     'output', 'corpus', 'corpus_tfidf2.mm'))  # 加载corpus_tfidf

    # 由于不需要根据corpus_tfidf的情况重新调整list_cut
    # 因此到此，Stage3完成。
    time_end3 = time.time()
    print('Stage3完成，用时：%.3f' % (time_end3-time_start3))
    print('                                        ')

    print('---------------------Stage4: 最佳模型筛选模块------------------------')
    time_start4 = time.time()

    # 由于不再像奥迪聆听项目那样需要对展示的主题数作一定的限制，因此主题数量的多少，
    # 本质上没有什么影响，反而数量多一些的主题的特征组合，对后面监督学习的结果会更好
    # 故，初步设定主题数量10个起步，step为5，limit为200.
    # 当然，后面limit的值可能会提高。为了方面在主函数中调参，将三个值均变量化，
    # 分别为initial_num_topics, limit_num_topics, step_num_topics
    # 并分1和2，1仍然代表content, 2代表title。
    # 分别进行content和title的建模模型
    # 积极评论内容(content_pos)建模--------------------------------------------------
    lmlist1_pos, c_v1_pos = evaluate_graph(
        dictionary1_pos, corpus_tfidf1_pos, list_cut1_1_pos_new, initial_num_topics1,
        limit_num_topics1, step_num_topics1, lda_param_minprob1,
        lda_param_iter1, lda_param_passes1, lda_param_chunksize1)

    # 如果人为需要查看，因此可以将产出模型的结果打印一下
    for m, cv in zip(range(initial_num_topics1,
                           limit_num_topics1, step_num_topics1), c_v1_pos):
        print("Content Num Topics =", m, " has Coherence Value of", round(cv, 4))

    tuple_topicnum_cv1_pos = zip(
        range(initial_num_topics1, limit_num_topics1, step_num_topics1), c_v1_pos)
    list_tuple_topicnum_cv1_pos = list(tuple_topicnum_cv1_pos)
    df_topicnum_cv1_pos = pd.DataFrame(list_tuple_topicnum_cv1_pos)
    df_topicnum_cv1_pos.to_csv(os.path.join(
        'output', 'table', 'pos_content_topicnum_cv.csv'))

    # 模型存储
    for i in range(len(lmlist1_pos)):
        lmlist1_pos[i].save(os.path.join('output', 'model_content_pos',
                                     'num_topic' + str((i + initial_num_topics1//step_num_topics1)*5) + '.model'))

    # # 通过algorithm的get_the_only_num_of_topcis得到合适的num_topics值
    best_model_num_topics1_pos = get_the_only_num_of_topcis(
        c_v1_pos, initial_num_topics1, limit_num_topics1, step_num_topics1, exceed_coef_denorminator, log_base)

    # # 打印一下最佳主题数量
    print('For content_pos, The best num_topics is: %d.' % best_model_num_topics1_pos)

    # 从存储的全部model中，取出最佳model
    model1_pos = models.ldamodel.LdaModel.load(os.path.join(
        'output', 'model_content_pos',
        'num_topic' + str(best_model_num_topics1_pos) + '.model'))

    # 与聆听项目不同的是，情感分析不关注词云的呈现效果，主要是需要文章与主题的分布概率，
    # （当然，后面还可能将词语与文章的分布概率作为补充特征，但目前不用）
    # 因此这里不需要使用algorithm中定制化程度高的top_topics_keywords_num函数
    # 对于top_topics的topn，取100即可。
    top_topics1_pos = model1_pos.top_topics(coherence='c_v', dictionary=dictionary1_pos,
                                   texts=list_cut1_1_pos_new,  topn=200)

    avg_topic_coherence1_pos = sum(
        [t[1] for t in top_topics1_pos]) / best_model_num_topics1_pos

    # 可以打印一下，看看主题的平均coherence_value是多少
    print('Content Average topic coherence: %.4f.' % avg_topic_coherence1_pos)

    # 将top_topics存储起来，需注意top_topics的数据结构
    top_topics_renew1_pos = []
    for top_topic, cv in top_topics1_pos:
        top_topics_renew1_pos.append(top_topic)

    top_topics_renew_df1_pos = pd.DataFrame(top_topics_renew1_pos)
    top_topics_renew_df1_pos.to_excel(os.path.join(
        'output', 'table', 'pos_content_top_topics_origin.xlsx'))

    # 消极评论内容(content_neg)建模----------------------------------------------------------------------------
    lmlist1_neg, c_v1_neg = evaluate_graph(
        dictionary1_neg, corpus_tfidf1_neg, list_cut1_1_neg_new, initial_num_topics1,
        limit_num_topics1, step_num_topics1, lda_param_minprob1,
        lda_param_iter1, lda_param_passes1, lda_param_chunksize1)

    # 如果人为需要查看，因此可以将产出模型的结果打印一下
    for m, cv in zip(range(initial_num_topics1,
                           limit_num_topics1, step_num_topics1), c_v1_neg):
        print("Content Num Topics =", m, " has Coherence Value of", round(cv, 4))

    tuple_topicnum_cv1_neg = zip(
        range(initial_num_topics1, limit_num_topics1, step_num_topics1), c_v1_neg)
    list_tuple_topicnum_cv1_neg = list(tuple_topicnum_cv1_neg)
    df_topicnum_cv1_neg = pd.DataFrame(list_tuple_topicnum_cv1_neg)
    df_topicnum_cv1_neg.to_csv(os.path.join(
        'output', 'table', 'pos_content_topicnum_cv.csv'))

    # 模型存储
    for i in range(len(lmlist1_neg)):
        lmlist1_neg[i].save(os.path.join('output', 'model_content_neg',
                                     'num_topic' + str((i + initial_num_topics1//step_num_topics1)*5) + '.model'))

    # # 通过algorithm的get_the_only_num_of_topcis得到合适的num_topics值
    best_model_num_topics1_neg = get_the_only_num_of_topcis(
        c_v1_neg, initial_num_topics1, limit_num_topics1, step_num_topics1, exceed_coef_denorminator, log_base)

    # # 打印一下最佳主题数量
    print('For content_neg, The best num_topics is: %d.' % best_model_num_topics1_neg)

    # 从存储的全部model中，取出最佳model
    model1_neg = models.ldamodel.LdaModel.load(os.path.join(
        'output', 'model_content_neg',
        'num_topic' + str(best_model_num_topics1_neg) + '.model'))

    # 与聆听项目不同的是，情感分析不关注词云的呈现效果，主要是需要文章与主题的分布概率，
    # （当然，后面还可能将词语与文章的分布概率作为补充特征，但目前不用）
    # 因此这里不需要使用algorithm中定制化程度高的top_topics_keywords_num函数
    # 对于top_topics的topn，取100即可。
    top_topics1_neg = model1_neg.top_topics(coherence='c_v', dictionary=dictionary1_neg,
                                   texts=list_cut1_1_neg_new,  topn=200)

    avg_topic_coherence1_neg = sum(
        [t[1] for t in top_topics1_neg]) / best_model_num_topics1_neg

    # 可以打印一下，看看主题的平均coherence_value是多少
    print('Content Average topic coherence: %.4f.' % avg_topic_coherence1_neg)

    # 将top_topics存储起来，需注意top_topics的数据结构
    top_topics_renew1_neg = []
    for top_topic, cv in top_topics1_neg:
        top_topics_renew1_neg.append(top_topic)

    top_topics_renew_df1_neg = pd.DataFrame(top_topics_renew1_neg)
    top_topics_renew_df1_neg.to_excel(os.path.join(
        'output', 'table', 'neg_content_top_topics_origin.xlsx'))

    # 问题内容（issue)建模----------------------------------------------------------------------------
    lmlist1_issue, c_v1_issue = evaluate_graph(
        dictionary1_issue, corpus_tfidf1_issue, list_cut1_1_issue_new, initial_num_topics1,
        limit_num_topics1, step_num_topics1, lda_param_minprob1,
        lda_param_iter1, lda_param_passes1, lda_param_chunksize1)

    # 如果人为需要查看，因此可以将产出模型的结果打印一下
    for m, cv in zip(range(initial_num_topics1,
                           limit_num_topics1, step_num_topics1), c_v1_issue):
        print("Content Num Topics =", m, " has Coherence Value of", round(cv, 4))

    tuple_topicnum_cv1_issue = zip(
        range(initial_num_topics1, limit_num_topics1, step_num_topics1), c_v1_issue)
    list_tuple_topicnum_cv1_issue = list(tuple_topicnum_cv1_issue)
    df_topicnum_cv1_issue = pd.DataFrame(list_tuple_topicnum_cv1_issue)
    df_topicnum_cv1_issue.to_csv(os.path.join(
        'output', 'table', 'pos_content_topicnum_cv.csv'))

    # 模型存储
    for i in range(len(lmlist1_issue)):
        lmlist1_issue[i].save(os.path.join('output', 'model_content_issue',
                                     'num_topic' + str((i + initial_num_topics1//step_num_topics1)*5) + '.model'))

    # # 通过algorithm的get_the_only_num_of_topcis得到合适的num_topics值
    best_model_num_topics1_issue = get_the_only_num_of_topcis(
        c_v1_issue, initial_num_topics1, limit_num_topics1, step_num_topics1, exceed_coef_denorminator, log_base)

    # # 打印一下最佳主题数量
    print('For content_issue, The best num_topics is: %d.' % best_model_num_topics1_issue)

    # 从存储的全部model中，取出最佳model
    model1_issue = models.ldamodel.LdaModel.load(os.path.join(
        'output', 'model_content_issue',
        'num_topic' + str(best_model_num_topics1_issue) + '.model'))

    # 与聆听项目不同的是，情感分析不关注词云的呈现效果，主要是需要文章与主题的分布概率，
    # （当然，后面还可能将词语与文章的分布概率作为补充特征，但目前不用）
    # 因此这里不需要使用algorithm中定制化程度高的top_topics_keywords_num函数
    # 对于top_topics的topn，取100即可。
    top_topics1_issue = model1_issue.top_topics(coherence='c_v', dictionary=dictionary1_issue,
                                   texts=list_cut1_1_issue_new,  topn=200)

    avg_topic_coherence1_issue = sum(
        [t[1] for t in top_topics1_issue]) / best_model_num_topics1_issue

    # 可以打印一下，看看主题的平均coherence_value是多少
    print('Content Average topic coherence: %.4f.' % avg_topic_coherence1_issue)

    # 将top_topics存储起来，需注意top_topics的数据结构
    top_topics_renew1_issue = []
    for top_topic, cv in top_topics1_issue:
        top_topics_renew1_issue.append(top_topic)

    top_topics_renew_df1_issue = pd.DataFrame(top_topics_renew1_issue)
    top_topics_renew_df1_issue.to_excel(os.path.join(
        'output', 'table', 'issue_content_top_topics_origin.xlsx'))

    # 标题内容(title)建模----------------------------------------------------------------------------
    lmlist2, c_v2 = evaluate_graph(
        dictionary2, corpus_tfidf2, list_cut2_4, initial_num_topics2,
        limit_num_topics2, step_num_topics2, lda_param_minprob2,
        lda_param_iter2, lda_param_passes2, lda_param_chunksize2)

    # 如果人为需要查看，因此可以将产出模型的结果打印一下
    for m, cv in zip(range(initial_num_topics2,
                           limit_num_topics2, step_num_topics2), c_v2):
        print("Title Num Topics =", m, " has Coherence Value of", round(cv, 4))

    tuple_topicnum_cv2 = zip(
        range(initial_num_topics2, limit_num_topics2, step_num_topics2), c_v2)
    list_tuple_topicnum_cv2 = list(tuple_topicnum_cv2)
    df_topicnum_cv2 = pd.DataFrame(list_tuple_topicnum_cv2)
    df_topicnum_cv2.to_csv(os.path.join(
        'output', 'table', 'title_topicnum_cv.csv'))

    # 模型存储
    for i in range(len(lmlist2)):
        lmlist2[i].save(os.path.join('output', 'model_title',
                                     'num_topic' + str((i + initial_num_topics2//step_num_topics2)*2) + '.model'))

    # 通过algorithm的get_the_only_num_of_topcis得到合适的num_topics值
    best_model_num_topics2 = get_the_only_num_of_topcis(
        c_v2, initial_num_topics2, limit_num_topics2, step_num_topics2, exceed_coef_denorminator, log_base)

    # 打印一下最佳主题数量
    print('For title, The best num_topics is: %d.' % best_model_num_topics2)

    # 从存储的全部model中，取出最佳model
    model2 = models.ldamodel.LdaModel.load(os.path.join(
        'output', 'model_title',
        'num_topic' + str(best_model_num_topics2) + '.model'))

    # 与聆听项目不同的是，情感分析不关注词云的呈现效果，主要是需要文章与主题的分布概率，
    # （当然，后面还可能将词语与文章的分布概率作为补充特征，但目前不用）
    # 因此这里不需要使用algorithm中定制化程度高的top_topics_keywords_num函数
    # 对于top_topics的topn，取100即可。
    top_topics2 = model2.top_topics(coherence='c_v', dictionary=dictionary2,
                                   texts=list_cut2_4,  topn=200)

    avg_topic_coherence2 = sum(
        [t[1] for t in top_topics2]) / best_model_num_topics2

    # 可以打印一下，看看主题的平均coherence_value是多少
    print('Content Average topic coherence: %.4f.' % avg_topic_coherence2)

    top_topics_renew2 = []
    for top_topic, cv in top_topics2:
        top_topics_renew2.append(top_topic)

    top_topics_renew_df2 = pd.DataFrame(top_topics_renew2)
    top_topics_renew_df2.to_excel(
        os.path.join('output', 'table', 'title_top_topics_origin.xlsx'))

    time_end4 = time.time()
    print('Stage4完成，用时：%.3f' % (time_end4-time_start4))
    print('                                                            ')

    print('-------------------Stage5: 模型结果输出模块--------------------')
    time_start5_3 = time.time()

    # 四个报表的输出。报表算法详见algorithm函数，首先是content_pos的报表1
    sent_topics_df1_pos = format_doc_dominant_topics(
        model1_pos, corpus_tfidf1_pos, list_cut1_1_pos_new, docs1)
    # 保存
    sent_topics_df1_pos.to_excel(os.path.join(
        'output', 'table', 'pos_content_df_dominant_topic.xlsx'))

    # 然后是neg的表1
    sent_topics_df1_neg = format_doc_dominant_topics(
    model1_neg, corpus_tfidf1_neg, list_cut1_1_neg_new, docs1)
    sent_topics_df1_neg.to_excel(os.path.join(
        'output', 'table', 'neg_content_df_dominant_topic.xlsx'))

    # issue的表1
    sent_topics_df1_issue = format_doc_dominant_topics(
    model1_issue, corpus_tfidf1_issue, list_cut1_1_issue_new, docs1)
    sent_topics_df1_issue.to_excel(os.path.join(
        'output', 'table', 'issue_content_df_dominant_topic.xlsx'))

    # title的报1同理
    sent_topics_df2 = format_doc_dominant_topics(
        model2, corpus_tfidf2, list_cut2_4, docs2)
    sent_topics_df2.to_excel(os.path.join(
        'output', 'table', 'title_df_dominant_topic.xlsx'))

    # 生成报表2，即文章与主题的关系矩阵，这步最为关键
    # 因为后面进行监督学习时，每篇文章的特征就是在各主题的分布概率
    doc_topic_ratio_df1_pos = matrix_table(model1_pos, corpus_tfidf1_pos, docs1)
    # 保存
    doc_topic_ratio_df1_pos.to_excel(os.path.join(
        'output', 'table', 'pos_content_matrix_docs_topics.xlsx'))

    # neg的表2
    doc_topic_ratio_df1_neg = matrix_table(model1_neg, corpus_tfidf1_neg, docs1)
    doc_topic_ratio_df1_neg.to_excel(os.path.join(
        'output', 'table', 'neg_content_matrix_docs_topics.xlsx'))

    # issue的表2
    doc_topic_ratio_df1_issue = matrix_table(model1_issue, corpus_tfidf1_issue, docs1)
    doc_topic_ratio_df1_issue.to_excel(os.path.join(
        'output', 'table', 'issue_content_matrix_docs_topics.xlsx'))

    # title的报表2
    doc_topic_ratio_df2 = matrix_table(model2, corpus_tfidf2, docs2)
    doc_topic_ratio_df2.to_excel(os.path.join(
        'output', 'table', 'title_matrix_docs_topics.xlsx'))

    # 生成报表3，即主题与关键词的矩阵
    df_top_topic_keywords1_pos = top_topic_words_df(top_topics1_pos)
    # 保存
    df_top_topic_keywords1_pos.to_excel(os.path.join(
        'output', 'table', 'pos_content_df_top_topic_keywords.xlsx'))

    # neg的表3
    df_top_topic_keywords1_neg = top_topic_words_df(top_topics1_neg)
    df_top_topic_keywords1_neg.to_excel(os.path.join(
        'output', 'table', 'neg_content_df_top_topic_keywords.xlsx'))

    # issue的表3
    df_top_topic_keywords1_issue = top_topic_words_df(top_topics1_issue)
    df_top_topic_keywords1_issue.to_excel(os.path.join(
        'output', 'table', 'issue_content_df_top_topic_keywords.xlsx'))

    # title的报表3
    df_top_topic_keywords2 = top_topic_words_df(top_topics2)
    df_top_topic_keywords2.to_excel(os.path.join(
        'output', 'table', 'title_df_top_topic_keywords.xlsx'))

    # 经过对报表4算法函数的进一步整合，在run()中可以用一个囊括了algorithm中其他
    # 函数的新函数将报表4导出
    # content的报表4
    df_topic_dominant_doc_final1_pos = df_topic_dominant_doc_table(model1_pos,
                                                               doc_topic_ratio_df1_pos,
                                                               list_cut1_1_pos_new,
                                                               docs1,
                                                               top_topics1_pos)
    # 保存
    df_topic_dominant_doc_final1_pos.to_excel(os.path.join(
        'output', 'table', 'pos_content_df_topic_dominant_doc_final.xlsx'))

    # neg的表4
    df_topic_dominant_doc_final1_neg = df_topic_dominant_doc_table(model1_neg,
                                                               doc_topic_ratio_df1_neg,
                                                               list_cut1_1_neg_new,
                                                               docs1,
                                                               top_topics1_neg)
    # 保存
    df_topic_dominant_doc_final1_neg.to_excel(os.path.join(
        'output', 'table', 'neg_content_df_topic_dominant_doc_final.xlsx'))

    # issue的表4
    df_topic_dominant_doc_final1_issue = df_topic_dominant_doc_table(model1_issue,
                                                               doc_topic_ratio_df1_issue,
                                                               list_cut1_1_issue_new,
                                                               docs1,
                                                               top_topics1_issue)
    # 保存
    df_topic_dominant_doc_final1_issue.to_excel(os.path.join(
        'output', 'table', 'issue_content_df_topic_dominant_doc_final.xlsx'))

    # title的报表4
    df_topic_dominant_doc_final2 = df_topic_dominant_doc_table(model2,
                                                               doc_topic_ratio_df2,
                                                               list_cut2_4,
                                                               docs2,
                                                               top_topics2)
    df_topic_dominant_doc_final2.to_excel(os.path.join(
        'output', 'table', 'title_df_topic_dominant_doc_final.xlsx'))

    time_end5_3 = time.time()
    print('Stage5-3报表输出完成，用时：%.3f' % (time_end5_3-time_start5_3))

    # time_start5_4 = time.time()

    # # 情感分析的主题划分，无需将高精度、低概率词拉到每个主题的最前面，
    # # 因此也无需应用algorithm中的topics_keywords_plus_tfidf，
    # # get_theme_wordcloud_dict函数进行top_topics的格式转化，
    # # 在draw_theme_wordcloud中直接设置topics = model.show_topics即可
    # # 但是在主题排序的时候最好还是注意一下排序，词云只是一个直观的查看方式
    # wordcloud_topics1_pos = top_topics_tuple_seq_adjust(top_topics1_pos)

    # draw_theme_wordcloud(dictionary1_pos, model1_pos,
    #                      wordcloud_topics1_pos, Folder_Path_0, 'pic_content_pos')

    # # neg的主题词云
    # wordcloud_topics1_neg = top_topics_tuple_seq_adjust(top_topics1_neg)

    # draw_theme_wordcloud(dictionary1_neg, model1_neg,
    #                      wordcloud_topics1_neg, Folder_Path_0, 'pic_content_neg')

    # # title的主题词云同理
    # wordcloud_topics2 = top_topics_tuple_seq_adjust(top_topics2)
    # draw_theme_wordcloud(dictionary2, model2,
    #                      wordcloud_topics2, Folder_Path_0, 'pic_title')

    # time_end5_4 = time.time()
    # print('Stage5-4重新绘制主题词云完成，用时：%.3f' % (time_end5_4-time_start5_4))

    # time_start5_5 = time.time()
    # # 最后一步，生成综合词云
    # origin_pic_topic_words_dict1_pos = no_theme_wordcloud_input(list_cut1_1_pos_new)
    # draw_integration_wordcloud(
    #     origin_pic_topic_words_dict1, Folder_Path_0, 'pic_content_pos')

    # # neg的综合词云
    # origin_pic_topic_words_dict1_neg = no_theme_wordcloud_input(list_cut1_1_neg_new)
    # draw_integration_wordcloud(
    #     origin_pic_topic_words_dict1, Folder_Path_0, 'pic_content_neg')

    # # 同理为title生成综合词云
    # origin_pic_topic_words_dict2 = no_theme_wordcloud_input(list_cut2_4)
    # draw_integration_wordcloud(
    #     origin_pic_topic_words_dict2, Folder_Path_0, 'pic_title')

    # time_end5_5 = time.time()
    # print('Stage5-5重新绘制综合词云完成，用时：%.3f' % (time_end5_5-time_start5_5))


if __name__ == "__main__":
    run('forum_0531_use')
