import pandas as pd
from str_num import is_number_regex, number_type_key
from utils import extract_keys, check_keys, save_to_csv, reverse_dependency_dict
from graph import build_graph_from_LLM_response, build_graph, remove_cycles, generate_dependencies
import networkx as nx
import matplotlib.pyplot as plt
from topological_sort import topological_layers_from_dag
from distribution import build_distribution_dict_for_str_features, build_precision_dict_for_num_features, pro_distribution_in_conditions, sample_from_dict
from gmm import estimate_gmm_probabilities, estimate_initiate_pro_distribution_for_num_features, safe_kde, estimate_conditional_distribution_KDE
from lookup import filter_rows_by_column_for_str_features, filter_rows_by_column_for_num_features
import time
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import gaussian_kde
import random
import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import BallTree
import concurrent.futures
        

dataset='house'
train_set_pth=dataset+"/train.csv"

### 第一步是处理数据，分出其中表示数字的内容和表示字符串的内容。
num_key, str_key=number_type_key(train_set_pth, verbose=True, blank="?")

descriptions={"adult":'A dataset for people income.',\
              "travel":"A Tour & Travels Company Wants To Predict Whether A Customer Will Churn Or Not Based On Indicators Given Below.",\
             "house":"The data pertains to the houses found in a given California district and some summary stats about them based on the 1990 census data."}

prompt= f'''
        I have a tabular dataset with the following description:
        "{descriptions[dataset]}"\n
        The dataset holds the following features, representing in numbers or text strings.
        {str(num_key+str_key)}\n
        Please list the constraints for each feature based on the others. Return the results in a specific format, which is: for each feature, first output the feature name followed by a colon, and then a set of constraints represented by square brackets. The '->' symbol indicates that the former is the cause and the latter is the effect. Different constraints should be separated by commas. 
        Here is an example:
        Feature A: [Feature B->Feature A, Feature C->Feature A]
        This means that both Feature B and Feature C determine the range of Feature A.
        You can leave a blank if there is no relation between Feature A and other features.
        Please give me the results directly, without any explaination.
        '''



response_income='''
age: []
fnlwgt: []
educational-num: [education->educational-num, age->educational-num]
capital-gain: [occupation->capital-gain, workclass->capital-gain, education->capital-gain, income->capital-gain]
capital-loss: [occupation->capital-loss, workclass->capital-loss, education->capital-loss, income->capital-loss]
hours-per-week: [occupation->hours-per-week, workclass->hours-per-week]
workclass: [occupation->workclass, income->workclass]
education: [educational-num->education]
marital-status: [age->marital-status, relationship->marital-status]
occupation: [education->occupation, age->occupation, workclass->occupation]
relationship: [marital-status->relationship, gender->relationship]
race: []
gender: []
native-country: []
income: [education->income, workclass->income, occupation->income, capital-gain->income, capital-loss->income, hours-per-week->income]
'''
response_travel='''
Age: []
ServicesOpted: [Age->ServicesOpted, AnnualIncomeClass->ServicesOpted]
Target: [ServicesOpted->Target, FrequentFlyer->Target, AnnualIncomeClass->Target]
FrequentFlyer: [Age->FrequentFlyer, AnnualIncomeClass->FrequentFlyer]
AnnualIncomeClass: [Age->AnnualIncomeClass, FrequentFlyer->AnnualIncomeClass]
AccountSyncedToSocialMedia: [Age->AccountSyncedToSocialMedia, ServicesOpted->AccountSyncedToSocialMedia]
BookedHotelOrNot: [Age->BookedHotelOrNot, ServicesOpted->BookedHotelOrNot]
'''
response_house='''
longitude: []
latitude: [longitude->latitude]
housing_median_age: [longitude->housing_median_age, latitude->housing_median_age, ocean_proximity->housing_median_age]
total_rooms: [longitude->total_rooms, latitude->total_rooms, population->total_rooms, households->total_rooms]
total_bedrooms: [longitude->total_bedrooms, latitude->total_bedrooms, total_rooms->total_bedrooms, population->total_bedrooms, households->total_bedrooms]
population: [longitude->population, latitude->population, total_rooms->population, households->population]
households: [longitude->households, latitude->households, total_rooms->households, population->households]
median_income: [longitude->median_income, latitude->median_income, ocean_proximity->median_income]
median_house_value: [longitude->median_house_value, latitude->median_house_value, housing_median_age->median_house_value, total_rooms->median_house_value, total_bedrooms->median_house_value, population->median_house_value, households->median_house_value, median_income->median_house_value, ocean_proximity->median_house_value]
ocean_proximity: [longitude->ocean_proximity, latitude->ocean_proximity]
'''


dependencies=build_graph_from_LLM_response(response, num_key+str_key)  #这里的是有环的
#print(dependencies)

G = build_graph(dependencies)

method='ilp' # pagerank ilp sc
DAG = remove_cycles(G, method)

# 生成新的 dependencies
dependencies = generate_dependencies(DAG)

dependencies={key:dependencies[key] for key in dependencies if key}

layers = topological_layers_from_dag(DAG, keys=num_key+str_key) # keys只是用来校验

### 第四步是统计现有数据集中的各个特征值的初始统计信息
df=pd.read_csv(train_set_pth)
str_pro_dis=build_distribution_dict_for_str_features(df, str_key, verbose=True) # 这里已经构建了字符串类的初始分布，字符串不会超出已有数据的类别

num_pre_dis=build_precision_dict_for_num_features(df, num_key, verbose=True) # 数值特征的最小值，最大值，变化区间

num_pro_dis=estimate_initiate_pro_distribution_for_num_features(df, num_key, num_pre_dis, verbose=True, n_components=3)

walk_graph = reverse_dependency_dict(dependencies)


dep_steps=2

if dep_steps>1:
    tmp_dependencies=dependencies
    for _ in range(dep_steps-1):
        for node in dependencies:
            for n in dependencies[node]:
                if n:
                    tmp_dependencies[node]+=dependencies[n]
            if "" in tmp_dependencies[node]:
                tmp_dependencies[node].remove("")
            if node in tmp_dependencies[node]:
                tmp_dependencies[node].remove(node)
            tmp_dependencies[node]=list(set(tmp_dependencies[node]))
        dependencies=tmp_dependencies




if SAMPLE_NUMS<=len(df):
    epoch_num=1
else:
    epoch_num=SAMPLE_NUMS//len(df)+1
    SAMPLE_NUMS=len(df)-1
    
Tree=False


if Tree:
    print("Building Ball Tree..")
    trees={}
    for key in num_key:
        trees[key] = BallTree(df[[key]].to_numpy(), leaf_size=40)  # 只用 target_key 这一列
    print("Finished.")
else:
    trees=None

print("Start Generating..")

def generate_sample(i):
    generated_sample = {}
    selected_sample = df.iloc[sample_indices[i]].to_dict()

    node = random.choice(layers[0])  # 从第一层中选择初始改变点
    if node in str_key:
        generated_value = sample_from_dict(str_pro_dis[node], top_p=None)
    elif node in num_key:
        generated_value = sample_from_dict(num_pro_dis[node], top_p=None)
    selected_sample[node] = generated_value

    nodes = walk_graph[node]
    while nodes:
        nodes_next_turn = []
        for node in nodes:
            conditional_keys = dependencies[node]

            CN = N
            max_count = 5
            c_count = 0
            while c_count < max_count:
                result = estimate_conditional_distribution_KDE(
                    df, num_key, str_key,
                    {key: selected_sample[key] for key in conditional_keys},
                    num_pre_dis, node, N=CN, tau=T, trees=trees,
                    method='kde', bw_method="silverman"                                  # 这里改KDE类型, scott
                )
                if result:
                    break
                else:
                    CN *= 2
                    c_count += 1
            if result:
                selected_sample[node] = sample_from_dict(result, top_p=None)

            nodes_next_turn += walk_graph[node]

        nodes = nodes_next_turn
    
    return selected_sample
    
generated_samples=[]

start_time = time.time()
for epoch in range(epoch_num):
    print(f"Epoch: {epoch}/{epoch_num}")
    sample_indices = random.sample(range(len(df)), SAMPLE_NUMS)
    # 使用 ThreadPoolExecutor 来异步处理任务
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        generated_samples = list(tqdm(executor.map(generate_sample, range(SAMPLE_NUMS)), total=SAMPLE_NUMS))

end_time = time.time()

print(f"Time cost: {(end_time - start_time) / SAMPLE_NUMS / epoch_num * 1000} ms.")


remove_training_repeat=False

tmp_l=[]
ori_l=[df.iloc[i].to_dict() for i in range(len(df))]
for sample in generated_samples:
    if sample not in tmp_l:
        if remove_training_repeat:
            if sample not in ori_l:
                tmp_l.append(sample)
        else:
            tmp_l.append(sample)
generated_samples=tmp_l
print(f"Remaining rate: {round(len(generated_samples)/SAMPLE_NUMS*100, 2)}")
print(f"Generation size: {len(generated_samples)}")

template=df.iloc[0].to_dict()
cleaned_outputs=[]
for sample in generated_samples:
    if sample:
        co={}
        for key in template:
            if key in num_key:
                if int(sample[key])==sample[key]:
                    co[key]=int(sample[key])
                else:
                    co[key]=sample[key]
            else:
                co[key]=sample[key]
        cleaned_outputs.append(co)
save_to_csv(cleaned_outputs, filename=dataset+"/"+"ILP_KDE_50_12.csv")

