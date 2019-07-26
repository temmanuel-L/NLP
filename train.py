import pandas as pd
import numpy as np
import sys
import os
import math
import time
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
from ETL_algorithm import *

# 该脚本用于对ETL生成的dataset进行XGBOOST训练，为了明晰训练流程的代码
# 将一些类似绘制ROC曲线的代码块定义成


def train(output_tryout_folder_name):
    # 为方便后面调试，仍然采用1st_try, 2nd_try, 3rd_try这样第n次的子文件夹名作为入参变量
    Folder_Path_0 = os.path.abspath(os.path.join(os.getcwd(), ".."))
    os.chdir(Folder_Path_0)

    print('--------------------Stage1: 导入并分割数据集--------------------')
    time_start1 = time.time()
    doc_emotion_set = pd.read_csv(os.path.join(
        'output_tryout', output_tryout_folder_name, 'train', 'dataset_concat.csv'))

    x_train, x_test, y_train, y_test = train_test_split(
        doc_emotion_set.iloc[:, 1:-1], pd.Series(doc_emotion_set.iloc[:, -1]),
        test_size=0.15, random_state=6)

    time_end1 = time.time()
    print('Stage1完成，用时：%.3f' % (time_end1-time_start1))
    print('                                                               ')
    # --------------------------------------------------------------------------
    print('-----------------------Stage2: 分类器训练-----------------------')
    time_start2 = time.time()

    model_emotion = XGBClassifier()
    model_emotion.fit(x_train, y_train)
    y_pred = pd.Series(model_emotion.predict(x_test))
    # xgb_params = {
    #     'learning_rate': 0.1,  # 步长
    #     'n_estimators': 20,
    #     'max_depth': 5,  # 树的最大深度
    #     'objective': 'multi:softmax',
    #     'num_class': 3,
    #     'min_child_weight': 1,  # 决定最小叶子节点样本权重和，如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。
    #     'gamma': 0,  # 指定了节点分裂所需的最小损失函数下降值。这个参数的值越大，算法越保守
    #     'silent': 0,  # 输出运行信息
    #     'subsample': 0.8,  # 每个决策树所用的子样本占总样本的比例（作用于样本）
    #     'colsample_bytree': 0.8,  # 建立树时对特征随机采样的比例（作用于特征）典型值：0.5-1
    #     'nthread': 4,
    #     'seed': 27}
    # print("training...")
    # model_emotion = xgb.train(xgb_params, xgb.DMatrix(x_train, y_train))
    # y_pred = model_emotion.predict(xgb.DMatrix(x_test))
    model_emotion_path = os.path.join('output_tryout', output_tryout_folder_name,
                                      'train', 'model_emotion.model')
    joblib.dump(model_emotion, model_emotion_path)

    accuracy = accuracy_score(y_test, y_pred)
    print("accuarcy: %.2f%%" % (accuracy*100.0))

    time_end2 = time.time()
    print('Stage2完成，用时：%.3f' % (time_end2-time_start2))
    print('                                                               ')
    # -------------------------------------------------------------------------
    print('------------------------Stage3: 结果输出------------------------')
    time_start3 = time.time()

    # 结果输出统一放在train文件夹下
    # 1. 绘制Feature_importance图片并保存
    fig, ax = plt.subplots(figsize=(10, 15))
    plot_importance(model_emotion, height=0.5, max_num_features=150, ax=ax)
    plt.savefig(os.path.join('output_tryout', output_tryout_folder_name,
                             'train',  'feature_importance.png'))
    plt.close()

    # 2. 打印classification_report
    # target_names_list = ['负面', '正面', '中性']

    # print(classification_report(y_test, y_pred,
    #                             target_names=target_names_list))
    print(classification_report(y_test, y_pred))
    # 该报告需要保存一下，为了将字符串保存成txt，先定义一个函数。
    # classification_report_txt = classification_report(
    #     y_test, y_pred, target_names=target_names_list)
    classification_report_txt = classification_report(
        y_test, y_pred)

    def string_save(filename, contents):
        fh = open(filename, 'w', encoding='utf-8')
        fh.write(contents)
        fh.close()

    string_save(os.path.join('output_tryout', output_tryout_folder_name,
                             'train', 'report.txt'), classification_report_txt)

    # 3. auc和ROC曲线
    y_score = model_emotion.predict_proba(x_test)

    y_one_hot = label_binarize(y_test, np.arange(3))
    print('调用函数auc：', metrics.roc_auc_score(
        y_one_hot, y_score, average='macro'))

    fpr, tpr, thresholds = metrics.roc_curve(
        y_one_hot.ravel(), y_score.ravel())
    auc = metrics.auc(fpr, tpr)
    print('手动计算auc：', auc)

    # 下面是绘制ROC曲线的代码
    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    # FPR就是横坐标,TPR就是纵坐标
    plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc)
    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    plt.title('情感模型三分类后的ROC和AUC', fontsize=17)
    plt.savefig(os.path.join('output_tryout', output_tryout_folder_name,
                             'train',  'ROC.png'))
    plt.close()

    time_end3 = time.time()
    print('Stage3完成，用时：%.3f' % (time_end3-time_start3))
    print('                                                               ')

    print('------------------------Stage4: 对原文的预测------------------------')
    time_start4 = time.time()

    # 用model_emotion对原文进行预测，故x_test应该修改为
    # doc_emotion_set.iloc[:, 1:-1]
    all_x_test = doc_emotion_set.iloc[:, 1:-1]
    all_y_pred = pd.Series(model_emotion.predict(all_x_test))
    all_y_pred_df = pd.DataFrame(all_y_pred)
    all_y_pred_df.columns = ['情感预测值']
    all_y_pred_df['情感预测值'] = all_y_pred_df['情感预测值'].astype("str")
    all_y_pred_df['情感预测值'] = all_y_pred_df['情感预测值'].apply(lambda x: reverse_emotion_words_transfer(x))
    origin_df = pd.read_excel(os.path.join('data_input', 'forum_0606.xlsx'))
    origin_add_y_pred = pd.concat([origin_df, all_y_pred_df], axis=1)

    origin_add_y_pred.to_excel(os.path.join(
        'output_tryout', output_tryout_folder_name, 'train', 'origin_add_y_pred.xlsx'))

    time_end4 = time.time()
    print('Stage4完成，用时：%.3f' % (time_end4-time_start4))
    print('                                                               ')


if __name__ == "__main__":
    train('5th_try')
