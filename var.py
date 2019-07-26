# 该文件主要作用调参的配置文件

# 主题模型的主题数量的起始值，10个起步
# 主题模型的主题数量的上限值，由于content内容非常庞杂，
# 故上限可以设置得高些，初步定为100个。
# 主题模型的主题数量的步进值，由于content的limit较大，
# 故步进也可大些，初步定为5。
initial_num_topics1 = 5
limit_num_topics1 = 50
step_num_topics1 = 5

# title的initial_num_topics, limit_num_topics, step_num_topics，
# 与content相比，显然是应该小一些的
initial_num_topics2 = 4    # 初定10个起步
limit_num_topics2 = 30    # 初定50个为上限
step_num_topics2 = 2        # 由于title的limit不大，因此步进也不能大，初定2

# 对于主题模型中的几个可能需要多次调整的超参数，包括：
# 1. minimum_probability 2. iterations 3. passes 4. chunksize
# 这四个超参数的预设值如下，同样是区分content和title的：
lda_param_minprob1 = 0.01
lda_param_iter1 = 200
lda_param_passes1 = 50
lda_param_chunksize1 = 100

lda_param_minprob2 = 0.1
lda_param_iter2 = 100
lda_param_passes2 = 20
lda_param_chunksize2 = 50


exceed_coef_denorminator = 10
log_base = 30
