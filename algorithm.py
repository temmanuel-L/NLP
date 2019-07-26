# 算法文档。
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
import pyLDAvis.gensim
import jieba
import ast
import time
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import matplotlib.colors as mcolors

# 该文档主要定义了train_process会用到的自定义函数，并写有注释说明。

# 接下来，作去除处理，在源码中，去除了单字母单词，以及数字（isnumeric()），唯有后者可能在汉语处理中用到


def check_and_mkdir(path_name_list):
    for path_name in path_name_list:
        mkdir_path = os.path.join('output', path_name)
        exist_judge = os.path.exists(mkdir_path)
        if not exist_judge:
            os.makedirs(mkdir_path)


def check_and_mkdir_predict(path_name_list):
    for path_name in path_name_list:
        mkdir_path = os.path.join('predict', path_name)
        exist_judge = os.path.exists(mkdir_path)
        if not exist_judge:
            os.makedirs(mkdir_path)


def stop_words(addr_stop_words):     # 该函数用于读入停用词
    stopwords = []
    with open(addr_stop_words) as f:
        stopwords = f.read()
    return stopwords


def text_list_cut_pkuseg(stopwords, text_list):
    # 采用pkuseg进行分词时使用
    # 该函数适用对象为总语料库，函数输出为双层列表。
    # 但更精准，用于一次性分词时使用
    list_cut = []
    seg = pkuseg.pkuseg(
        model_name=r'C:\Users\mathartsys\Anaconda3\Lib\site-packages\pkuseg\models\pku_ctb8_weibo_model')
    for text in text_list:
        text_cut = [word for word in seg.cut(text)]
        text_cut_new = []
        for word in text_cut:
            if word not in stopwords:
                text_cut_new.append(word)
        list_cut.append(text_cut_new)
    return list_cut


def text_list_cut_jieba(stopwords, text_list):
    # 采用jieba进行分词
    # 该函数适用对象为总语料库，函数输出为双层列表。
    # 但更精准，用于一次性分词时使用
    list_cut = []
    for text in text_list:
        text_cut = [word for word in jieba.cut(str(text))]
        text_cut_new = []
        for word in text_cut:
            if word not in stopwords:
                text_cut_new.append(word)
        list_cut.append(text_cut_new)
    return list_cut


def text_cut_pkuseg(stopwords, text):
    # 该函数同样采用pkuseg分词
    # 适用对象为单一文本，比如进行测试的一个文本。因为其中只有一层列表
    seg = pkuseg.pkuseg(
        r'C:\Users\mathartsys\Anaconda3\Lib\site-packages\pkuseg\models\pku_ctb8_weibo_model')
    text_cut = [word for word in seg.cut(text)]
    text_cut_new = []
    for word in text_cut:
        if word not in stopwords:
            text_cut_new.append(word)
    return text_cut_new


def text_cut_jieba(stopwords, text):
    # 该函数同样采用jieba分词
    # 适用对象为单一文本，比如进行测试的一个文本。因为其中只有一层列表
    text_cut = [word for word in jieba.cut(text)]
    text_cut_new = []
    for word in text_cut:
        if word not in stopwords:
            text_cut_new.append(word)
    return text_cut_new


def get_only_list_words_from_docs(list_cut, special_list):
    # 该函数主要用于快速从已初步完成切词的list_cut（两层列表）中
    # 取出在白名单的词语，这一目的是为了服务于情感分析，避免正、负面
    # 词语过多地在同一主题出现，以致于作为后续监督学习特征的主题并没有
    # 进行情绪分离的作用。则这样训练出来的XGBOOST模型的准确率必然不高。
    list_cut_new = []
    for doc in list_cut:
        doc_new = []
        for word in doc:
            if word in special_list:
                doc_new.append(word)
        list_cut_new.append(doc_new)
    return list_cut_new


def get_positive_words_list():
    # 获得下面情绪词列表，以作为正面主题的正向过滤字典
    df_pos = pd.read_excel(os.path.join('dict', 'pos.xlsx'))
    pos_list = df_pos.pos_name.tolist()
    return pos_list


def get_negative_words_list():
    # 获得负面情绪词列表，以作为负面主题的正向过滤字典
    df_neg = pd.read_excel(os.path.join('dict', 'neg.xlsx'))
    neg_list = df_neg.neg_name.tolist()
    return neg_list


def get_issue_words_list():
    df_issue = pd.read_excel(os.path.join('dict', 'issue_words_adjust.xlsx'))
    issue_list = df_issue.issue_name.tolist()
    return issue_list


def get_sentiment_mark_words_list():
    # 获得情感分值词表，以作为正向过滤字典
    df_sentiment_mark = pd.read_csv(os.path.join(
        'dict', 'userdict_sentiment_mark.csv'), encoding='gbk')
    sentiment_mark_list = df_sentiment_mark.sentiment_mark.tolist()
    return sentiment_mark_list


def sentiment_mark_count(sentiment_mark_dict, list_cut):
    # 该函数主要解决句子内各词情感值加和的问题
    docs_sentiment_mark_list = []
    for doc in list_cut:
        doc_sentiment_mark = 0
        for word in doc:
            if word in sentiment_mark_dict.keys():
                doc_sentiment_mark += sentiment_mark_dict[word]
        docs_sentiment_mark_list.append(doc_sentiment_mark)
    return docs_sentiment_mark_list


def add_word_combination(list_cut1, min_count):
    # 两个入参，list_cut1为黑白名单加入后的切词结果，min_count为拟新建的组合词语的最少出现次数。
    bigram = Phrases(list_cut1, min_count=min_count)
    list_cut1_new = list_cut1
    for idx in range(len(list_cut1_new)):
        for words in bigram[list_cut1_new[idx]]:
            if '_' in words:
                # 只有当出现了_的新词组时，才将新词组加入到list_cut1_new中
                list_cut1_new[idx].append(words)
    return list_cut1_new


def drop_number(list_cut1_new):
    # list_cut1_new，切记列表，为双层嵌套列表
    list_cut2 = [[word for word in doc if not word.isnumeric()]
                 for doc in list_cut1_new]
    return list_cut2


def special_regexp_list():
    # 该函数不需要入参，将定义一系列非过滤指定汉字的表达式，并返回集合列表。
    # 时间、日期正则查询的第一种,即数字+文字，数字在1-4位，文字为1位。
    regexp_dt_1 = re.compile(r'^\d{1,4}[\u4E00-\u9FA5]+$')

    # 时间、日期正则查询第2种，“数字-数字-数字”，且每个数字位数不超过4位。数字之间的连接符有5种可能。
    # 有一种特殊的情况是2019.1.15号，即在最后的数字后又跟了字符，为此，最后再配0+个汉字或单词字符，正则表达式如下：
    regexp_dt_2 = re.compile(
        r'^\d{1,4}[-./:：]\d{1,4}[-./:：]\d{1,4}[\u4E00-\u9FA5]*\w*$')

    # 时间、日期正则查询第3种，即“数字：数字”，且每个数字位数不超2位
    regexp_dt_3 = re.compile(r'^\d{1,}\D{1}\d{1,}$')

    # # 时间、日期正则查询第4种，即“数字 + 年 + 数字 + 月 + 数字 + 日”，且每个数字位数不超过2位。
    regexp_dt_4 = re.compile(
        r'^\d{1,4}[\u4E00-\u9FA5]*\d{1,4}[\u4E00-\u9FA5]*\d{1,4}[\u4E00-\u9FA5]*$')

    # 其他第1种，如：LFV3A24G9，即“字母{1,}+数字{1,}+字母+数字+字母+数字”的方式，为了使其更有扩展性，
    # 从第3个字母开始，所有的匹配次数都是0+，即*，但同时，为了防止将车型这种有意义的字母+数字的字符串找出来，
    # 可作如下泛化，即头部为字母{1，}，数字{1，}，尾部为3个以上的单词字符。
    regexp_other_1 = re.compile(r'^[A-Za-z]+[0-9]+\w{3,}$')

    # 其他第2种，如：-----，这种过长的非单词字符,至少多长算作过长，暂且拍脑袋定为5个字符以上。
    regexp_other_2 = re.compile(r'^\W{7,}$')

    # 其他第2_1种，如：aaaaaaaaaa, 1111111111111这种过长的单词字符，暂且定为10个以上的连续
    regexp_other_2_1 = re.compile(r'^[A-Za-z0-9]{8,}$')

    # 车牌第1种，如：苏EAW77T，浙A2YX58，即车牌号信息，一个汉字后跟了6个单词字符，为保险起见，我们设为5-8个。
    regexp_carplate_1 = re.compile(r'^[\u4E00-\u9FA5]{1}\w{5,8}$')

    # 车牌第2种，因jieba会把车牌的汉字头部去除，故需要新增一个去除汉字头部的正则查询，实例为：'A2YX58'
    regexp_carplate_2 = re.compile(r'^[A-Za-z]{1}[A-Za-z0-9]{5}$')

    # 其他第3种，去除连续两个以上字母开头的情况.
    regexp_other_3 = re.compile(
        r'^[A-Za-z]{2,}[0-9]{0,}[A-Za-z]{0,}[0-9]{0,}[A-Za-z]{0,}[0-9]{0,}$')

    # 其他第4种，该正则情况较为复杂，其原因在于引入了双语组合词，而组合词之间统一使用"_"连接，这就会导致原来已经分开的如5月，2日，
    # 组合成了新的词组“5月_2日”，为了针对所有的组合词语，正则规则设定如下：如果连接符两端未都紧跟汉字，则去除
    regexp_other_4 = re.compile(r'^.*[\u4E00-\u9FA5]{1,}_\d{1,}.*$')

    # 其他第5种，与其他第4种呼应，第4种是"_"前为汉字，第5种为"_"后是汉字
    regexp_other_5 = re.compile(r'^.*\w{1,}_[\u4E00-\u9FA5]{1,}.*$')

    # 其他第6种，即非汉字、非字母、非数字情况，即非实体的字符
    regexp_other_6 = re.compile(r'^[^\u4e00-\u9fa5A-Za-z]{1,}$')

    # 对于重复出现的汉字，比如：艾艾、好好、聪聪这样的双同字词语，往往是表示名字的，可过滤
    regexp_name_1 = re.compile(r'^.*([\u4E00-\u9FA5])\1{1,}.*$')

    # 对于包含“先生”的词语，需去除
    regexp_name_2 = re.compile(r'^.*[\u5148][\u751F].*$')

    # 对于包含“女士”的词语，需要去除
    regexp_name_3 = re.compile(r'^.*[\u5973][\u58EB].*$')

    # 对于包含“小姐“的词，需要去除
    regexp_name_4 = re.compile(r'^.*[\u5C0F][\u59D0].*$')

    # # 对于包含“奥迪”字样的词语，也应该去除
    # regexp_audi_1 = re.compile(r'^.*[\u5965][\u8FEA].*$')

    regexp_special_list = [regexp_dt_1, regexp_dt_2, regexp_dt_3, regexp_dt_4, regexp_carplate_1, regexp_carplate_2,
                           regexp_other_1, regexp_other_2, regexp_other_2_1, regexp_other_3, regexp_other_4, regexp_other_5, regexp_other_6,
                           regexp_name_2, regexp_name_3, regexp_name_4]

    return regexp_special_list


def char_utf_df_processing(char_utf_df, screened_words_new):
    # 用于处理导入的char_utf_df(汉字、UTF码对应表)，入参为char_utf-df, screened_words_new，即一个对比表，一个汉字表，以求得码表。
    # 首先将\xao转成''
    char_utf_new = []
    for x in char_utf_df.values:
        x_new = []
        for y in x:
            y = y.replace('\xa0', '')
            x_new.append(y)
        char_utf_new.append(x_new)
    # 新建一个空列表，以存储utf码，效果如下：['4E00','4E03','4E09','4E5D','4E8C','4E94','516B','516D',……]
    utf_screen_list = []
    for i in range(len(char_utf_new)):
        if char_utf_new[i][3] in screened_words_new:
            utf_screen_list.append(char_utf_new[i][1])
    return utf_screen_list


def general_regexp_list(utf_screen_list):
    #  新建函数，用于返回泛化的正则规则列表，列表中每个表达式只找到包含一个指定汉字的所有词
    regexp_general_list = []
    for j in range(len(utf_screen_list)):
        regexp_screen_char = re.compile(
            r'^.*' + r'[\u' + utf_screen_list[j] + r']' + r'{1,}.*$')
        regexp_general_list.append(regexp_screen_char)
    return regexp_general_list


################################################################################################################
# 下面几个函数一体的，最终功能是将得到的多个正则表达式，在list_cut2中完成过滤操作。
def regexp_find_results_for_list(regexp, doc):
    # 建一个函数，以实现每个正则查询的结果
    result_list = []
    for x in doc:
        result = regexp.findall(x)
        if result:
            result_list.append(x)
    return result_list


def regexp_result_flatten(input_list):
    # 上面函数的result已经是一个列表，故result_list是一个二级列表，需要对其展开为一级列表，为此，再定义一个flatten函数，
    # 以将二级列表展开为一级列表。
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


def regexp_flatten_batch_processing(regexp_list, doc):
    # 至此，output_list就是每一次查询展开后的一级列表，但是当定义的正则查询项不断增多时，对每个查询都执行上面两个函数非常繁琐，
    # 故建立一个执行函数，在其中调用上面两个函数。
    # regexp_list是已定义好的正则查询的列表
    list_regexp_result_flatten = []
    for regexp in regexp_list:
        reg_result = regexp_find_results_for_list(regexp, doc)
        reg_result_flatten = regexp_result_flatten(reg_result)
        list_regexp_result_flatten.append(reg_result_flatten)
    return list_regexp_result_flatten


def regexp_result_flatten_union(list_regexp_result_flatten):
    # 所有查询的并集才是我们想去除的全部内容，求得并集后，将用list_cut2与其求差集,
    # 结果自然就是过滤器想要达到的效果，首先定义正则的并集函数
    # 由于使用到了前两个函数，因此
    # regexp_result_union用来存储正则查询结果的并集，先将list_regexp_result的第一项赋值给它
    regexp_result_union = list_regexp_result_flatten[0]
    for i in range(1, len(list_regexp_result_flatten), 1):
        regexp_result_union = list(set(regexp_result_union).union(
            set(list_regexp_result_flatten[i])))
    return regexp_result_union


def list_cut_screened_regexp(list_cut, regexp_list):
    # list_cut_screened是最终的执行函数，会包括上面的函数。入参为list_cut(要过滤的对象), regexp_list(定义的正则查询列表)
    # 在list_cut遍历所有doc，每个doc都其各自的regexp_result_union求差集
    # 在此之前，新建一个空列表，用来存储过滤正则查询后的新的list_cut。
    list_cut_screened = []
    for doc in list_cut:
        list_regexp_result_flatten = regexp_flatten_batch_processing(
            regexp_list, doc)

        # 调用regexp_result_flatten_union，以对查询结果求并集
        regexp_result_union = regexp_result_flatten_union(
            list_regexp_result_flatten)

        # 对每个doc都与regexp_result_union求差集。
        doc_screened = list(set(doc).difference(set(regexp_result_union)))
        list_cut_screened.append(doc_screened)

    # 最后返回list_cut_screened，就是过滤后可直接建立词袋的新的list_cut。
    return list_cut_screened
##################################################################################################


def delete_len_one_words(list_cut):
    # 对于分词结果，若词语长度为1，则删去
    list_cut_new = []
    for doc in list_cut:
        doc_new = []
        for word in doc:
            if len(word) > 1:
                doc_new.append(word)
        list_cut_new.append(doc_new)
    return list_cut_new


def other_to_be_screened_list(addr_address, addr_firstname,
                              addr_hf_normal, addr_dealer):
    # 过滤器之开源字典
    # 入参即为四个开源数据字典，分别为地址、姓式、高频无用词、文本团队的黑名单
    # 再新增两个入参，为经销商和车型字典地址
    # 函数的目的是将其整理化列表化，以方面与待筛选列表求差集
    address_str = pd.read_excel(addr_address)
    address_to_be_screened_list = address_str.place_name.tolist()

    firstname_str = pd.read_excel(addr_firstname)
    firstname_to_be_screened_list = firstname_str.NameB.tolist()

    hf_normal_str = pd.read_excel(addr_hf_normal)
    hf_normal_to_be_screened_list = hf_normal_str.wordname.tolist()

    dealer_str = pd.read_excel(addr_dealer)
    dealer_to_be_screened_list = dealer_str.name.tolist()

    return address_to_be_screened_list, firstname_to_be_screened_list, \
        hf_normal_to_be_screened_list, dealer_to_be_screened_list


def list_cut_screened_dict(list_cut, list_dict):
    # 将list_cut2中的每个子列表分别与上面三个表求差集，另外，需要注意的是，为了加快速度，开源库的筛选还是应该在正则之后，
    # 而如果做完开源库的筛选，发现正则规则还需要进行补充，则可直接重新运行正则部分。
    list_cut_new = []
    for m in range(len(list_cut)):
        doc_diff = list(set(list_cut[m]).difference(set(list_dict[0])))
        for i in range(1, len(list_dict), 1):
            doc_diff = list(set(doc_diff).difference(set(list_dict[i])))
        list_cut_new.append(doc_diff)
        # print(m)

    return list_cut_new


def corpus_tfidf_list_new(corpus_tfidf):
    # 原来的corpus_tfidf仍然是全量文章的空间向量，虽然是已经过滤掉很多词语，
    # 但为了进一步优化词云效果，可考虑只取每篇文章的top20个词
    corpus_tfidf_list = list(corpus_tfidf)
    corpus_tfidf_list_new = []
    for doc_vec in corpus_tfidf_list:
        doc_vec_new = []
        doc_vec_sort = sorted(doc_vec, key=lambda x: (x[1]), reverse=True)
        doc_vec_new = doc_vec_sort[:20]
        corpus_tfidf_list_new.append(doc_vec_new)

    return corpus_tfidf_list_new


def list_cut_reform_accord_to_corpus_tfidf_new(corpus_tfidf_new, dictionary):
    # 按照corpus_tfidf_new的结果，重新构造list_cut2，因为后面list_cut和corpus都需要入模LDA模型。
    corpus_tfidf_new_to_words = []
    for x in corpus_tfidf_new:
        doc_vec_new = []
        for m, n in x:
            word_name = dictionary[m]
            doc_vec_new.append(word_name)
        corpus_tfidf_new_to_words.append(doc_vec_new)

    return corpus_tfidf_new_to_words


def list_cut_reform(list_cut2, corpus_tfidf_new_to_words):
    # 将list_cut2与list_cut_new求交集,得到重构后的list_cut2
    list_cut2_new = []
    for i in range(len(list_cut2)):
        doc_new = list(set(list_cut2[i]) & set(corpus_tfidf_new_to_words[i]))
        list_cut2_new.append(doc_new)

    return list_cut2_new


#############################################################################################
# 下面是建模函数
# 该CELL的目的，主要是tweak一下num_topics的最佳个数
def evaluate_graph(dictionary, corpus_tfidf_new, texts,
                   start, limit, step, lda_param_minprob, lda_param_iter,
                   lda_param_passes, lda_param_chunksize):
    # 最重要的一个函数，定义主要的超参数，并参数不同模型数量，交叉验证出效果最好的模型
    chunksize = lda_param_chunksize
    passes = lda_param_passes
    iterations = lda_param_iter
    eval_every = None
    # 这一步很关键，如果不设置的话，后面生成topic, doc矩阵时，会发现有些topic，对于有些文章其贡献非常小，
    # 在保留4位小数时小的不可见，这样也就没办法找到结果了。
    # 还需要补充一点，对于情感分析的主题模型训练，尤其是针对短文本的情况，
    # 因为多数短文本的只有两三句话，少的就一句话或几个字，因此minimum_probability
    # 这个参数此时就显示出威力来了。如果认为短文本评论的句子数众数为2，则
    # minimum_probability应为0.5，但如此划分必须会分出数量众多的主题，这样主题的
    # 数量便会非常接近评论的数量，主题便起不到降维和鲁棒性的作用了。
    # 当然一句话只提高了一个主题，不等于主题的贡献率为100%，这是两个概念。
    # 暂且将其设成0.1，后面可能需要针对minimum_probability进行大量的调整
    # 为此，有必要将其放入变量配置文件var.py中，以进行快速调整。
    # 同理，iterations, passes, chunksize也需进入配置文件。
    minimum_probability = lda_param_minprob
    # 设置两个空列表，一个填充不同num_topic下的coherence_value, 另一个填充不同num_topic下的模型结果
    c_v = []
    lm_list = []

    for num_topics in range(start, limit, step):
        lm = models.ldamodel.LdaModel(corpus=corpus_tfidf_new, num_topics=num_topics, id2word=dictionary, alpha='auto', eta='auto',
                                      iterations=iterations, passes=passes, eval_every=eval_every,
                                      random_state=6, minimum_probability=minimum_probability)
        lm_list.append(lm)
        cm = models.CoherenceModel(
            model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())
        print(num_topics)

    # 显示不同num_topics的模型的coherence打分结果趋势图
    x = range(start, limit, step)
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("c_v"), loc='best')
    # plt.show()

    return lm_list, c_v


def get_the_only_num_of_topcis(c_v, initial_num_topics, limit_num_topics, step_num_topics,
                               exceed_coef_denorminator, log_base):
    # 用于自动化得到最佳主题数量
    # 在经历了几次尝试之后，我们对问题作进一步简化，不再考虑波峰、和最适用主题数范围等因素，
    # 直接采用主题数与coherence_value之间的关系作为唯一的判定标准。
    # 先在第一轮内进行内部过滤，每轮都按主题数对元组进行升序排列，求每个元组的主题数与第一个元组主题数量之比，是一个比值的列表。
    # 以列表中ni设定每个元组的及格线系数，1/3(lgn)，用第一个元组的coherence_value，乘以该系数，得到每个元组应该达到的及格分
    # 对于超过及格分的元组，append进入一个新的元组列表，判定一下该列表中的元组数量，如果为0，其他元组相比于第一个元组，均无法达到
    # 及格分，则最佳主题数为第一个元组的[0]，如果数量大于等于1，则说明其他元组有超过及格分的，为此，再重复上面的操作，直到append的
    # 新列表内数量为0为至。
    topicnum_cv_list = list(
        zip(range(initial_num_topics, limit_num_topics, step_num_topics), c_v))

    # topicnum_cv_list的内容为元组，元组无法修改，故将其所有层级列表化
    topicnum_cv_list_process = topicnum_cv_list
    topicnum_cv_list_1 = []
    for m, n in topicnum_cv_list_process:
        topicnum_cv_list_2 = []
        topicnum_cv_list_2.append(m)
        topicnum_cv_list_2.append(n)
        topicnum_cv_list_1.append(topicnum_cv_list_2)

    while len(topicnum_cv_list_1) >= 1:
        # 循环定义一个空列表，while循环终止后，保存终止前留下的最后赢家
        the_chosen_one = []
        the_chosen_one.append(topicnum_cv_list_1[0])

        for x in topicnum_cv_list_1:
            basic_mark = topicnum_cv_list_1[0][1]   # 动态定义及格分基数
            basic_topicnum = topicnum_cv_list_1[0][0]  # 动态定义主题基数
            num_topic_ratio = float(x[0])/basic_topicnum    # 动态倍数
            exceed_coef = 1 + (1/exceed_coef_denorminator) * \
                math.log(num_topic_ratio, log_base)  # 及格分系数
            exceed_mark = topicnum_cv_list_1[0][1]*exceed_coef  # 及格分
            # 通过对比x[1]与exceed_mark的大小，分别让x添加0或1
            if x[1] <= exceed_mark:
                x.append(0)
            else:
                x.append(1)

        # 将x[2]为1的选出，这里我们通过dataframe的方式达成
        topicnum_cv_list_1_df = pd.DataFrame(topicnum_cv_list_1)
        topicnum_cv_list_1_df_screened = topicnum_cv_list_1_df[topicnum_cv_list_1_df[2] == 1]
        # 再将topicnum_cv_list_df_screened转回列表
        topicnum_cv_list_1 = [list(x)
                              for x in topicnum_cv_list_1_df_screened.values]

        # 在一次while循环后，减少了元素数量的topic_cv_list_1还会有append新增exceed_mark的操作
        # 因此，需要在下一次while循环之前，将topic_cv_list_1的exceed_mark去除
        for x in topicnum_cv_list_1:
            x.remove(x[2])

    # while循环结束前的最后一次的the_chosen_one是一个双层列表
    # 则最佳主题数即为the_chosen_one[0][0], 对应的coherence_value是the_chosen_one[0][1]
    # 另外，由于主题数在前面需要进行除法，转换为了float，而存储的model名字里数字是整数转成的字符串
    # 故该函数最后返回的也必须是整数。
    return int(the_chosen_one[0][0])


def top_topics_keywords_num(dictionary, model):
    # 该函数用来选择最佳主题中的参数topn，仍然以round(len(dictionary)/(10 * model.num_topics))
    # 作为判断依据,返回变量top_topics_param_topn，以给后面的model.top_topics的topn参数赋值
    # if round(len(dictionary)/(10 * model.num_topics)) < 80:
    if model.num_topics > 15:
        top_topics_param_topn = 180
    elif model.num_topics > 9:
        top_topics_param_topn = 200
    else:
        top_topics_param_topn = 250

    return top_topics_param_topn

############################################################################################
# 下面是词云相关函数，在调取wordcloud函数之前，需要对top_topics的数据形式进行调整。


def top_topics_tuple_seq_adjust(top_topics):
    # 但由GENSIM内置函数输出的top_topics的格式与show_topics的格式不一致，为此需要调整top_topics的格式。
    top_topics_seq_adjust = []
    for top_topic, cv in top_topics:
        top_topic_seq_adjust = pd.DataFrame()
        for prop, word in top_topic:
            top_topic_seq_adjust = top_topic_seq_adjust.append(
                pd.Series([word, round(prop, 5)]), ignore_index=True)
        # 转化成dataframe之后的top_topic_seq_adjust，再通过.values函数转成调整顺序的元组
        top_topic_seq_adjust_tuple = [tuple(xi)
                                      for xi in top_topic_seq_adjust.values]
        top_topics_seq_adjust.append(top_topic_seq_adjust_tuple)
    return top_topics_seq_adjust


def subplots_row_col_selection(model):
    # 该函数用于下面的词云构建函数，主要是由主题数量自动判断采用哪种框架画图合适
    if model.num_topics >= 12:
        fig, axes = plt.subplots(3, 4, figsize=(
            20, 20), sharex=True, sharey=True)
    elif model.num_topics >= 9:
        fig, axes = plt.subplots(3, 3, figsize=(
            20, 20), sharex=True, sharey=True)
    elif model.num_topics >= 6:
        fig, axes = plt.subplots(2, 3, figsize=(
            20, 20), sharex=True, sharey=True)
    elif model.num_topics >= 4:
        fig, axes = plt.subplots(2, 2, figsize=(
            30, 30), sharex=True, sharey=True)
    elif model.num_topics >= 3:
        fig, axes = plt.subplots(1, 3, figsize=(
            30, 30), sharex=True, sharey=True)

    return fig, axes


def draw_theme_wordcloud(dictionary, model,
                         wordcloud_topics, Folder_Path_0, pic_son_folder):
    # 该函数直接绘制主题词云
    # 有三个入参，其中前两个主要是为了更合适地动态呈现关键词的数量，第三个入参是上面函数的出参。
    # 新增一个入参，为动态子目录，以适合对多问题数据的动态输出。
    # 首先从XKCD_COLORS中获得色彩RGB值，TABLEAU_COLORS的色彩只有十个
    cols = [color for name, color in mcolors.XKCD_COLORS.items()]

    # if round(len(dictionary)/(10 * model.num_topics)) >= 80:
    if model.num_topics > 12:
        num_max_words = 80
    else:
        num_max_words = 100

    my_wordcloud = WordCloud(scale=2,
                             background_color='white',
                             width=3000,
                             height=2500,
                             max_words=num_max_words,
                             #                   colormap='tab30',
                             color_func=lambda *args, **kwargs: cols[i],
                             prefer_horizontal=1.0,
                             font_path=r'C:\Windows\Fonts\msyh.ttc',
                             random_state=666)

    # topics = lda_model.show_topics(formatted=False)

    # 利用subplots_row_col_selection，选取合适的画图框架
    fig, axes = subplots_row_col_selection(model)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(wordcloud_topics[i])
        my_wordcloud.generate_from_frequencies(topic_words, max_font_size=400)

        plt.rcParams['savefig.dpi'] = 300  # 图片像素
        plt.rcParams['figure.dpi'] = 300  # 分辨率

        plt.gca().imshow(my_wordcloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    os.chdir(Folder_Path_0)
    plt.savefig(os.path.join('output', pic_son_folder, 'theme_cloud.png'))
    plt.close()


def no_theme_wordcloud_input(list_cut2_new):
    # 主题模型词云图之后，可以再呈现一个无主题的综合词云，该函数输出这个词云的字典输入
    # 因此输入为最新处理的list_cut2_new。
    dictionary_list_cut2_new = corpora.Dictionary(list_cut2_new)

    # 用Dictionary的dfs呈现词序与词频的关系
    origin_pic_topic_words = dictionary_list_cut2_new.dfs

    # 新建一个空字典，用来存储即将生成的综合词云的输入字典
    origin_pic_topic_words_dict = {}
    for word_no, freq in origin_pic_topic_words.items():
        word_name = dictionary_list_cut2_new[word_no]
        origin_pic_topic_words_dict[word_name] = freq

    return origin_pic_topic_words_dict


def draw_integration_wordcloud(origin_pic_topic_words_dict,
                               Folder_Path_0, pic_son_folder):
    my_wordcloud = WordCloud(background_color="white", width=4000, height=3000,
                             font_path="C:/Windows/Fonts/msyh.ttc").generate_from_frequencies(
                                 origin_pic_topic_words_dict, max_font_size=400)
    plt.imshow(my_wordcloud)
    plt.axis("off")
    os.chdir(Folder_Path_0)
    plt.savefig(os.path.join('output', pic_son_folder, 'concat_cloud.png'))
    plt.close()


#####################################################################################################
# 下面是最后一部分，报表的生成函数
def format_doc_dominant_topics(ldamodel, corpus, list_cut2_new, docs):
    # 首先以每个文章为对象，输出最佳主题名称
    # 三个入参分别为：使用的模型、语料库（传参时用corpus_tfidf）、list_cut2_new（总之选最新的那个list_cut）
    # 定义一个空的dataframe
    sent_topics_df = pd.DataFrame()

    # 给每篇文章加上概率最大的主题。
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if (j == 0):  # row倒序排列后，排在最顶上的就是这个doc最可能的topic
                wp = ldamodel.show_topic(topic_num, topn=50)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(i), int(topic_num), round(prop_topic, 4), topic_keywords]),
                                                       ignore_index=True)
            else:
                break
    sent_topics_df.columns = [
        'Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Topic_Keywords']

    contents1 = pd.DataFrame(pd.Series(list_cut2_new))
    contents1.reset_index(inplace=True)
    sent_topics_df = pd.merge(sent_topics_df, contents1, how='left',
                              left_on='Document_No', right_on='index').drop('index', axis=1)

    contents2 = pd.DataFrame(pd.Series(docs))
    contents2.reset_index(inplace=True)
    sent_topics_df = pd.merge(sent_topics_df, contents2, how='left',
                              left_on='Document_No', right_on='index').drop('index', axis=1)
#     sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    # 目前sent_topics_df共有6列，最后两列没有列名，需要定义列名
    sent_topics_df.columns = ['Document_No', 'Dominant_Topic',
                              'Topic_Perc_Contrib', 'Topic_Keywords', 'Text_Cut', 'Text_Origin']
    return sent_topics_df


def format_topic_dominant_docs(df_dominant_topic, doc_no_col, dom_topic_num_col_name, dom_topic_perc_col_name):
    # 这个函数就是在上面函数的基础上group by，这个函数意义不大，因为大部分主题的贡献都不大，不能成为一篇文章的最大贡献主题
    # 因此这样只能选出少数的普遍性主题，而且这些普遍性主题往往都是top_topics中排名靠后的主题
    # 四个传参，依次为：1. 上面函数生成的df_dominant_topic表格，2. 文章序号列的列名，3.主题序号列的列名，4.概率列的列名
    # 先定义一个空的dataframe
    df_dominant_doc = pd.DataFrame()

    # 由于第一传参df_dominant_topics里面出现了documnet_no，不利于生成以Dominant_Topic为group by对象的列表，故删去第一入参的'Document_No'列
    del df_dominant_topic[doc_no_col]

    # 对第一传参，即表格df_dominant_topic，按照其dominant_topic列名进行group by
    df_dominant_topic_grouped = df_dominant_topic.groupby(
        dom_topic_num_col_name)

    for i, group in df_dominant_topic_grouped:
        df_dominant_doc = pd.concat([df_dominant_doc, group.sort_values(
            [dom_topic_perc_col_name], ascending=[0]).head(1)], axis=0)

    # 重置一下索引
    df_dominant_doc.reset_index(drop=True, inplace=True)

    # 编辑一下列名
    df_dominant_doc.columns = [
        'Dominant_Topic', "Topic_Perc_Contrib", "Topic_Keywords", "Text_Cut", "Text_Origin"]

    return df_dominant_doc


def format_topic_dominant_statistics(df_dominant_topic, df_dominant_doc, topic_num_col, topic_keywords_col):
    # 最后一个函数，给出每个topic涉及的文章数等统计参数，这在pyLDAvis显示的效果中也能看出来，能是通过列表展示，结果更加量化
    # 该报表呈现意义也不大，因为是在表1和表2的基础上生成的，还是显示的为普遍性表格
    # 主题数量统计,由于要用到value_counts(),所以需要使用第一个表
    topic_counts = df_dominant_topic[topic_num_col].value_counts()
    df_counts = pd.DataFrame(topic_counts)
    df_counts.reset_index(inplace=True)
    df_counts.columns = [topic_num_col, 'counts']

    # 每个主题的文章概率
    df_perc = pd.DataFrame((round(topic_counts/topic_counts.sum(), 4)))
    df_perc.reset_index(inplace=True)
    df_perc.columns = [topic_num_col, 'perc']

    # 主题序号和关键词,注意这里要用生成的第二个表，即df_dominant_doc
    topic_num_keywords = df_dominant_doc[[topic_num_col, topic_keywords_col]]

    # 四列合一
    df_dominant_topics_statistics = pd.merge(
        topic_num_keywords, df_counts, on=topic_num_col, how='left')
    df_dominant_topics_statistics = pd.merge(
        df_dominant_topics_statistics, df_perc, on=topic_num_col, how='left')

    # 重命名列名
    df_dominant_topics_statistics.columns = [
        'Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

    return df_dominant_topics_statistics


def matrix_table(model, corpus_tfidf_new, docs):
    # 该表比表1重要，是呈现所有文章和主题对应概率关系的矩阵
    # 三个入参分别是主程序中的model, corpus_tfidf_new和docs
    # 先用产生的ldamodel加载一下corpus_tfidf_new（这里选corpus_tfidf_new，而不是corpus_tfidf）
    lda_output = model[corpus_tfidf_new]

    doc_topic_ratio_all_list = []
    for doc_topic_ratio in list(lda_output):
        doc_topic_ratio_all = doc_topic_ratio    # 先给其赋值。
        topic_index_for_doc = []
        for m, n in doc_topic_ratio:
            topic_index_for_doc.append(int(m))
        for i in range(model.num_topics):
            if i not in topic_index_for_doc:
                doc_topic_ratio_all.append((i, 0))
        # 这时的doc_topic_ratio_all中每个元组的首位是不按顺序排列的，需重新排列
        doc_topic_ratio_all_sort = sorted(
            doc_topic_ratio_all, key=lambda x: x[0], reverse=False)
        # 当然，再后面转为dataframe时，每个元素不能为元组，而需要是唯一的值，为此，将doc_topic_ratio_all_sort列表元组中的概率取出
        doc_topic_ratio_all_sort_notuples = []
        for m, n in doc_topic_ratio_all_sort:
            doc_topic_ratio_all_sort_notuples.append(n)
        doc_topic_ratio_all_list.append(doc_topic_ratio_all_sort_notuples)

    # 生成矩阵的列名列表
    topicnames = ["Topic" + str(i) for i in range(model.num_topics)]

    # 生成矩阵的行名列表
    docnames = ["Doc" + str(i) for i in range(len(docs))]

    # 生成dataframe
    doc_topic_ratio_df = pd.DataFrame(
        doc_topic_ratio_all_list, columns=topicnames, index=docnames)

    # 在矩阵的最后一列，新增一个每篇文章最普遍的主题序号列
    dominant_topic_series = np.argmax(doc_topic_ratio_df.values, axis=1)
    doc_topic_ratio_df['dominant_topic'] = dominant_topic_series

    return doc_topic_ratio_df


def top_topic_words_df(top_topics):
    # 该图是最重要的报表，将词云报表化
    # 前者的图用于识别关键信息，后者的图用于识别大贡献率信息，由于大贡献率信息本身也较为常见，应为主机厂业务部门所
    # 熟知，因此top_topics，即coherence_score高的主题更有意义
    # 定义一个函数，用来将top_topics转化为list，以方便进一步转化为dataframe.
    top_topic_words_list = []
    test_i = 0
    for i in range(len(top_topics)):
        top_topic_words = []
        try:
            for word_cv, word_name in top_topics[i][0]:
                top_topic_words.append(word_name)
        except:
            pass
        top_topic_words_list.append(top_topic_words)

    df_top_topic_keywords = pd.DataFrame(top_topic_words_list)
    df_top_topic_keywords.columns = [
        'Word '+str(i) for i in range(df_top_topic_keywords.shape[1])]
    df_top_topic_keywords.index = [
        'CV_Topic '+str(i) for i in range(df_top_topic_keywords.shape[0])]

    return df_top_topic_keywords


#######################################################################################################
# 下面是最后一个报表函数，由于相对复杂，会涉及多个函数
def get_dominant_doc_for_each_topic(model, doc_topic_ratio_df):
    # 该函数是最后一个报表的入参函数，用来生成每个主题对应的概率最大文章号的索引列表。
    perc_contrib_list = []     # 新建list, 用于存放各主题对应的最大概率值
    doc_index_list = []     # 新建list, 用于存放文章序号
    for j in range(model.num_topics):
        df_test = pd.DataFrame()   # 新建一个dataframe，用于分组使用
        for i, group in doc_topic_ratio_df.groupby('Topic' + str(j)):
            df_test = pd.concat([df_test, group.sort_values(
                ['Topic' + str(j)], ascending=[0]).head(1)], axis=0)
        df_test = df_test.sort_values('Topic'+str(j), ascending=False)
        perc_contrib_list.append(df_test.iloc[0, j].round(3))
        # 至此，df_test.iloc[0, j]就是j主题下贡献率最大的文章的概率值，接下来得到该文章序号
        df_test.reset_index(inplace=True)
        index_string = df_test['index'][0]
        regexp_doc_index = re.compile(r'\d{1,}')
        doc_index_number_list = regexp_doc_index.findall(index_string)
        # doc_index_number就是j主题下对应的贡献最大的文章序号
        doc_index_number = int(doc_index_number_list[0])
        doc_index_list.append(doc_index_number)
        # print(j)

    # 完成完备主题的概率、文章序号的列表收集后，分别将两个列表转化成series,再合并成dataframe
    perc_contrib_series = pd.Series(perc_contrib_list)
    doc_index_series = pd.Series(doc_index_list)
    df_perc_index = pd.concat([perc_contrib_series, doc_index_series], axis=1)

    # 返回df_perc_index
    return df_perc_index


def reformat_topic_dominant_docs(model, df_perc_index, list_cut2_new, docs):
    # 该函数以df_perc_index为入参，通过df_perc_index默认的索引关联topic的keywords, 通过doc_index关联listcut和doc_origin
    df_perc_index_new = df_perc_index.reset_index()
    # 如此，df_perc_index便成了三列，将其转为元组，方便进行dataframe的append一行的操作。
    # 注意，这个tuples包含三个值，建立循环的时候使用topic_index, topic_perc_contrib, doc_index
    df_perc_index_tuples = [tuple(xi) for xi in df_perc_index_new.values]

    # 新建一个空的dataframe，用于存储要存储的值，主要就是topic_index, topic_keywords, perc, doc_index
    re_df_topic_dominant_doc = pd.DataFrame()

    # 为topic_index_list匹配topic_keywords
    for topic_index, topic_perc_contrib, doc_index in df_perc_index_tuples:
        wp = model.show_topic(int(topic_index), topn=30)
        topic_keywords = ','.join([word for word, prop in wp])
        re_df_topic_dominant_doc = re_df_topic_dominant_doc.append(pd.Series([int(topic_index), topic_keywords,
                                                                              round(
                                                                                  topic_perc_contrib, 3),
                                                                              int(doc_index)]), ignore_index=True)

    re_df_topic_dominant_doc.columns = [
        'Topic_No', 'Topic_Keywords', 'Topic_Perc_Contrib', 'Document_No']

    # 在上面四列的基础上，通过Document_No关联list_cut和origin_doc信息
    contents1 = pd.DataFrame(pd.Series(list_cut2_new))
    contents1.reset_index(inplace=True)
    re_df_topic_dominant_doc = pd.merge(re_df_topic_dominant_doc, contents1,
                                        how='left', left_on='Document_No', right_on='index').drop('index', axis=1)

    contents2 = pd.DataFrame(pd.Series(docs))
    contents2.reset_index(inplace=True)
    re_df_topic_dominant_doc = pd.merge(re_df_topic_dominant_doc, contents2,
                                        how='left', left_on='Document_No', right_on='index').drop('index', axis=1)

    re_df_topic_dominant_doc.columns = [
        'Topic_No', 'Topic_Keywords', 'Topic_Perc_Contrib', 'Document_No', 'Doc_cut', 'Doc_Origin']
    trans_to_int_list = ['Topic_No', 'Document_No']

    for col in trans_to_int_list:
        re_df_topic_dominant_doc[col] = re_df_topic_dominant_doc[col].astype(
            "int")
    # re_df_topic_dominant_doc有六列信息，将其返回
    return re_df_topic_dominant_doc


def get_keywords_feature(top_topics):
    list1 = []
    for top_topic, topic_cv in top_topics:
        list2 = []
        for prop, word in top_topic:
            list2.append(word)
        list1.append(list2)
    return list1


def top_topics_match_slice(get_keywords_feature):
    # 先建立由top_topics抽出的keywords匹配切片函数
    top_topics_match_slice = []
    for x in get_keywords_feature:
        top_topics_match_slice.append(x[:8])
    return top_topics_match_slice


def topic_dominant_doc_match_slice(re_df_topic_dominant_doc, keywords_col):
    test_series = re_df_topic_dominant_doc[keywords_col].tolist()
    topic_dom_doc_match_slice = []
    for x in test_series:
        y = x.split(',')
        y_new = y[:8]
        topic_dom_doc_match_slice.append(y_new)
    return topic_dom_doc_match_slice


def df_topic_dominant_doc_table(model, doc_topic_ratio_df, list_cut, docs,
                                top_topics):
    # 由于生成表4需要付出很多步骤，在run()函数中将步骤列出会显得非常繁琐，
    # 为了让run()流程显得更清晰，建立该函数，将上面几个功能单一的分函数串起来，
    # 以最终输出表4
    df_perc_index = get_dominant_doc_for_each_topic(model,
                                                    doc_topic_ratio_df)

    df_perc_index.columns = ['Topic_Perc_Contrib', 'Doc_Index']

    re_df_topic_dominant_doc = reformat_topic_dominant_docs(
        model, df_perc_index, list_cut, docs)

    get_keywords_feature_var = get_keywords_feature(top_topics)

    # 执行切片操作
    top_topics_match_slice_var = top_topics_match_slice(
        get_keywords_feature_var)

    topic_dominant_doc_match_slice_var = topic_dominant_doc_match_slice(
        re_df_topic_dominant_doc, 'Topic_Keywords')

    # 以top_topics_match_slice为基准，建立dataframe，
    # 而且由于切片列已变为了列表，需再将其变回string
    df_top_topics_match_slice = pd.DataFrame(
        pd.Series(top_topics_match_slice_var), columns=['top_topics_slice'])

    df_top_topics_match_slice['top_topics_slice'] = df_top_topics_match_slice['top_topics_slice'].astype(
        "str")

    # 以topic_dominant_doc_match_slice为基准，建立dataframe
    df_topic_dominant_doc_match_slice = pd.DataFrame(pd.Series(
        topic_dominant_doc_match_slice_var), columns=['topic_dominant_doc_slice'])

    re_df_topic_dominant_doc_new = pd.concat(
        [re_df_topic_dominant_doc, df_topic_dominant_doc_match_slice], axis=1)

    re_df_topic_dominant_doc_new['topic_dominant_doc_slice'] = re_df_topic_dominant_doc_new['topic_dominant_doc_slice'].astype(
        "str")

    # 再以df_top_topics_match_slice为左基准，把re_df_topic_dominant_doc_new合并进来
    df_topic_dominant_doc_final = pd.merge(df_top_topics_match_slice, re_df_topic_dominant_doc_new, how='left',
                                           left_on='top_topics_slice', right_on='topic_dominant_doc_slice').drop('topic_dominant_doc_slice', axis=1)

    del df_topic_dominant_doc_final['top_topics_slice']

    return df_topic_dominant_doc_final


##############################################################################
# 为了从主题关键词中进一步将客户关心的词语排到前面，我们需要用TFIDF对主题关键词进行筛选和排序
# 那么，选取什么样的语料库的TFIDF？经过多轮试验，发现TFIDF的语料库效果越大越好，为此，建立
# big_corpus.py文件，以生成全文档的语料库的TFIDF，即corpus_big_tfidf和dictionary_big
# 然后通过下面的算法，对TFIDF的重要性区分作进一步的两极分化
def corpus_big_tfidf_flat_sort(corpus_big_tfidf):
    # 该函数将corpu_big_tfidf中每篇文章的所有词向量都放到一个列表里，即从双层列表下降为一层列表
    # 以进行按词序的groupby，以对词权重作进一步分化处理
    word_vec_concat_list = []
    for doc_vec in corpus_big_tfidf:
        for word_vec in doc_vec:
            word_vec_concat_list.append(word_vec)

    # 将展开后的一维列表，按词序由小到大排序
    word_vec_concat_list_sort = sorted(
        word_vec_concat_list, key=lambda x: (x[0]), reverse=False)

    return word_vec_concat_list_sort


def corpus_big_tfidf_flat_sort_process(word_vec_concat_list_sort, dictionary_big):
    # 以上面函数的出参作为入参，主要进行groupby, 对同一词序求weight的mean()，再将mean()/dictionary.dfs
    # 通过以上变换，得到每个词序的最终weight
    df_groupby = pd.DataFrame(word_vec_concat_list_sort)
    df_groupby.columns = ['word_no', 'tfidf_value']

    # 将groupby的结果求均值
    word_tfidf_adjust = df_groupby.groupby(['word_no']).mean()

    # 再将word_tfidf_adjust变换成dataframe,主要是方便转元组
    word_tfidf_adjust_reset = word_tfidf_adjust.reset_index()

    # dataframe转元组
    word_tfidf_adjust_list = [tuple(xi)
                              for xi in word_tfidf_adjust_reset.values]

    # 将word_tfidf_adjust_list中每个词序的weight,分别除以其词频，结果作为新的weight
    # 新建一个空列表，以存储这些新的weight.
    list_word_tfidf_processed = []
    for word_no, word_tfidf_mean in word_tfidf_adjust_list:
        word_tfidf_processed = word_tfidf_mean/dictionary_big.dfs[word_no]
        # word_tfidf_processed = word_tfidf_mean
        list_word_tfidf_processed.append(word_tfidf_processed)

    # 将其转成Series
    series_word_tfidf_processed = pd.Series(list_word_tfidf_processed)

    # 最后将上面的series与word_tfidf_adjust_reset这个dataframe合并，主要是为了关联word_tfidf_adjust_reset中的词序
    word_tfidf_adjust_concat = pd.concat(
        [word_tfidf_adjust_reset, series_word_tfidf_processed], axis=1)

    # 将合并后的dataframe起列名
    word_tfidf_adjust_concat.columns = [
        'word_no', 'tfidf_value', 'tfidf_value_processed']

    # 从新的dataframe中选取词序列和新的词权重列
    word_tfidf_adjust_processed = word_tfidf_adjust_concat[[
        'word_no', 'tfidf_value_processed']]

    # 再转元组
    word_tfidf_adjust_processed_list = [
        tuple(xi) for xi in word_tfidf_adjust_processed.values]

    # 倒序排列
    word_tfidf_adjust_processed_list_sort = sorted(
        word_tfidf_adjust_processed_list, key=lambda x: x[1], reverse=True)

    # word_tfidf_adjust_processed_list_sort显示的是词序号的权重，需要将词序号通过大词袋dictionary_big转成词语
    # 则list_test_all显示的就是[词名，词权重]的形式
    list_test_all = []
    for m, n in word_tfidf_adjust_processed_list_sort:
        list_test_one = []
        list_test_one.append(dictionary_big[m])
        list_test_one.append(n)
        list_test_all.append(list_test_one)

    # 需要将list_test_all转成元组形式，即(词，词权重)
    list_test_all_df = pd.DataFrame(list_test_all)
    list_test_all_tuple = [tuple(x) for x in list_test_all_df.values]

    # 返回list_test_all_tuple，这就是全文档语料库给出效果更好的每个词在人为操纵下的新权重，
    # 用该列表与主题的keywords进行交叉对比，即可得到最终结果。
    return list_test_all_tuple


def topics_keywords_plus_tfidf(df_top_topic_keywords, list_test_all_tuple):
    # 该函数就是将topics中的keywords, 重新以上面函数生成的list_test_all_tuple中的词权重作为重新排序依据。
    # 入参两个：1. 输出结果的主题——关键词矩阵表， 2. 上面函数的出参
    num_topics = df_top_topic_keywords.shape[0]
    list_tuple_screened_all = []
    for i in range(num_topics):
        this_topic_keywords = df_top_topic_keywords.iloc[i, :].tolist()
        list_tuple_screened = []
        for m, n in list_test_all_tuple:
            list_one_tuple_screened = []
            if m in this_topic_keywords:
                list_one_tuple_screened.append(m)
                list_one_tuple_screened.append(n)
                list_tuple_screened.append(list_one_tuple_screened)
        list_tuple_screened_all.append(list_tuple_screened)

    return list_tuple_screened_all


def get_theme_wordcloud_dict(list_tuple_screened_all):
    # list_tuple_screened_all与跟可以直接入参主题画图的wordcloud_topics形式还不一样
    # 后者是列表嵌套元组的形式，故对list_tuple_screened_all形式稍作调整
    list_tuple_screened_all_renew = []
    for topic in list_tuple_screened_all:
        topic_df = pd.DataFrame(topic)
        topic_df_tuple = [tuple(xi) for xi in topic_df.values]
        list_tuple_screened_all_renew.append(topic_df_tuple)

    return list_tuple_screened_all_renew


def get_composite_wordcloud_keywords(list_tuple_screened_all):
    # 入参就是topics_keywords_plus_tfidf()的出参。
    list_tuple_flat = []
    for topic in list_tuple_screened_all:
        for x in topic:
            list_tuple_flat.append(x)

    # 为了元组化，还是先转成dataframe
    list_tuple_flat_df = pd.DataFrame(list_tuple_flat)
    list_tuple_flat_df.columns = ['keywords', 'weight']
    # 元组化之前，重新考虑一下累积权重问题，因为topics_keywords_plut_tfidf已经能把
    # 客户关心的词排在前面了，因此如果采用groupby后的sum()求各关键词的权重，显然区分度会更好
    list_tuple_flat_df_groupby = list_tuple_flat_df.groupby(['keywords']).sum()

    list_tuple_flat_df_groupby_reset = list_tuple_flat_df_groupby.reset_index()

    # 执行元组化
    keywords_for_one_issue_concat = [
        tuple(xi) for xi in list_tuple_flat_df_groupby_reset.values]

    # 倒序排列，其实意义不大，因为词云函数会自行判定。
    keywords_for_one_issue_concat_sort = sorted(
        keywords_for_one_issue_concat, key=lambda x: x[1], reverse=True)

    return keywords_for_one_issue_concat_sort

##############################################################################################
# 下步的函数是为准备上线时新增的


def build_corpus_for_new_docs(dictionary, list_cut):
    new_doc_corpus = []
    for doc_cut in list_cut:
        corpus_for_one_doc = dictionary.doc2bow(doc_cut)  # 2.转换成bow向量
        new_doc_corpus.append(corpus_for_one_doc)

    return new_doc_corpus


def build_corpus_tfidf_for_new_docs(corpus, new_doc_corpus):
    tfidf = models.TfidfModel(corpus)
    new_doc_corpus_tfidf = []
    for item in new_doc_corpus:
        item_tfidf = tfidf[item]
        new_doc_corpus_tfidf.append(item_tfidf)
    return new_doc_corpus_tfidf
