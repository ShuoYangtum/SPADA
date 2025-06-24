from collections import defaultdict, namedtuple
from concurrent.futures import ThreadPoolExecutor
from distribution import build_distribution_dict_for_str_features, build_precision_dict_for_num_features, pro_distribution_in_conditions, sample_from_dict
from gmm import estimate_gmm_probabilities, estimate_initiate_pro_distribution_for_num_features, safe_kde, estimate_conditional_distribution_KDE
from graph import build_graph_from_LLM_response, build_graph, remove_cycles, generate_dependencies
from joblib import Parallel, delayed
from lookup import filter_rows_by_column_for_str_features, filter_rows_by_column_for_num_features
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import AffineCouplingTransform, ReversePermutation, CompositeTransform
from nf import SwiGLU, ContextNet, nearest_valid_value, predict 
import pandas as pd
import random
from str_num import is_number_regex, number_type_key
from scipy.stats import gaussian_kde
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import BallTree
from sklearn.linear_model import BayesianRidge
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from topological_sort import topological_layers_from_dag
from utils import extract_keys, check_keys, save_to_csv, reverse_dependency_dict

class SPADA:
    def __init__(self, patient=500, epoch=5000):
        self.early_stop=True
        self.patient=patient
        self.epoch=epoch
        self.bw_method="silverman"
    def fit_prompt(self, description, num_key, str_key):
        prompt= f'''
        You are given a tabular dataset described as follows:
        f{description}
        
        The dataset contains the following features (numerical or categorical):
        f{num_key + str_key}
        
        Your task is to identify the constraints (feature dependencies) between features. For each target feature, list the features that determine or constrain it, using the following format:
        
        Feature_Name: [Cause1->Feature_Name, Cause2->Feature_Name]
        
        Use '->' to indicate a constraint (i.e., one feature constrains or determines another). If a feature is independent, return an empty list:
        Feature_Name: []
        
        Only return the constraint list for each feature. Do not include any explanation or additional text.
        '''
        return prompt
    def generate_feature_map(self, csv_file_path, target_feature):
        try:
            # 读取CSV文件的第一行以获取表头
            df = pd.read_csv(csv_file_path, nrows=0)
            all_features = df.columns.tolist()
        except FileNotFoundError:
            return f"错误：文件 '{csv_file_path}' 未找到。"
        except Exception as e:
            return f"读取CSV文件时发生错误：{e}"
    
        if target_feature not in all_features:
            return f"错误：目标特征 '{target_feature}' 在CSV文件中不存在。"
    
        feature_map_lines = []
        for feature in all_features:
            if feature == target_feature:
                # 目标特征的列表包含所有其他特征指向它的箭头
                other_features = [f"{f}->{target_feature}" for f in all_features if f != target_feature]
                feature_map_lines.append(f"{target_feature}: [{', '.join(other_features)}]")
            else:
                # 其他特征的列表为空
                feature_map_lines.append(f"{feature}: []")
    
        return '\n'.join(feature_map_lines)

    def fit(
        self, 
        data,
        description,
        response=None,
        blank='?',
        target_feature=None,
        Tree=True):

            
        num_key, str_key=number_type_key(data, verbose=True, blank=blank)
        prompt=self.fit_prompt(description, num_key, str_key)
            
        if target_feature:
            response=self.generate_feature_map(data, target_feature)
        # generate a response from prompt
        response='\n'.join([i for i in response.split('\n') if i])
        dependencies=build_graph_from_LLM_response(response, num_key+str_key)  

        G = build_graph(dependencies)

        DAG = remove_cycles(G, 'ilp')
        dependencies = generate_dependencies(DAG)

        dependencies={key:dependencies[key] for key in dependencies if key}
        layers = topological_layers_from_dag(DAG, keys=num_key+str_key) 
        self.layers=layers
        df=pd.read_csv(data)
        self.df=df
        str_pro_dis=build_distribution_dict_for_str_features(df, str_key, verbose=True) 

        num_pre_dis=build_precision_dict_for_num_features(df, num_key, verbose=True) 
        num_pro_dis=estimate_initiate_pro_distribution_for_num_features(df, num_key, num_pre_dis, verbose=True, n_components=3)

        walk_graph = reverse_dependency_dict(dependencies)
        self.walk_graph=walk_graph
        early_stop=self.early_stop
        patient=self.patient
        epoch_num=self.epoch
        Interleaved_Mask=False
        ModelWrapper = namedtuple("ModelWrapper", ["type", "model"])
        
        df_flow=df.copy()
        target_features = [k for k, v in dependencies.items() if v and k in num_key]
        
        print(target_features)
        
        scalers = {}
        encoders = {}
        for feature in num_key:
            scaler = StandardScaler()
            df_flow[feature] = scaler.fit_transform(df_flow[[feature]])
            scalers[feature] = scaler

        for feature in str_key:  
            encoder = LabelEncoder()
            df_flow[feature] = encoder.fit_transform(df_flow[feature])  
            encoders[feature] = encoder 
            
        trained_models = {}
        
        for target in target_features:
            if early_stop:
                b_loss=float("inf")
                count=0
            conditions = dependencies[target]
            conditions = [c for c in conditions if c in df_flow.columns] 
        
            if not conditions:
                continue
        
            X = df_flow[conditions].values
            Y = df_flow[[target]].values
            
            if len(conditions) == 1:
                X = df_flow[conditions].values
                Y = df_flow[[target]].values
                model = make_pipeline(PolynomialFeatures(10), BayesianRidge())
                
                model.fit(X, Y.ravel())
                print(f"{target} | Gaussian fitting.")
                trained_models[target] = ModelWrapper("gaussian", model)
            else:
        
                X = torch.tensor(X, dtype=torch.float32)
                Y = torch.tensor(Y, dtype=torch.float32)
            
        
                base_dist = StandardNormal(shape=(1,))
            
                num_features=X.shape[1]
                if Interleaved_Mask:
        
                    mask = torch.tensor([1, 0] * (num_features // 2) + [1] * (num_features % 2))
                    transforms = [
                        ReversePermutation(features=1),
                        AffineCouplingTransform(
                            mask=mask, 
                            transform_net_create_fn=lambda in_features, out_features: ContextNet(in_features, out_features, context_features=X.shape[1])
                        )
                    ]
                else:
                    transforms = [
                        ReversePermutation(features=1),
                        AffineCouplingTransform(mask=torch.tensor([1]), \
                                                transform_net_create_fn=lambda in_features, out_features: \
                                                                            ContextNet(in_features, out_features, context_features=X.shape[1])
                                                )
                                ]
                flow = Flow(CompositeTransform(transforms), base_dist)
        
                optimizer = torch.optim.AdamW(flow.parameters(), lr=5e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            
                for epoch in range(epoch_num):
                    optimizer.zero_grad()
                    log_prob = flow.log_prob(Y, context=X)
                    loss = -log_prob.mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                        
                    if float(loss.item()) < b_loss:
                        b_loss = float(loss.item())
                        count = 0
                    else:
                        count += 1
                        if early_stop and count >= patient:
                            print(f"Early stopped at epoch {epoch}, best loss: {b_loss}")
                            break
                
                    if epoch % 100 == 0:
                        print(f"{target} | Epoch {epoch}, Loss: {loss.item()}")
        
                trained_models[target] = ModelWrapper("flow", flow)
        self.trained_models=trained_models
        self.str_pro_dis=str_pro_dis
        self.num_pro_dis=num_pro_dis
        self.dependencies=dependencies
        self.num_key=num_key
        self.str_key=str_key
        self.num_pre_dis=num_pre_dis
        self.scalers=scalers
        self.encoders=encoders
        if Tree:
            print("Building Ball Tree..")
            trees={}
            for key in num_key:
                trees[key] = BallTree(df[[key]].to_numpy(), leaf_size=40)  
            print("Finished.")
        else:
            trees=None

        self.trees=trees
        
    def generate_sample(self, i):
        bw_method=self.bw_method
        num_key=self.num_key
        str_key=self.str_key
        str_pro_dis=self.str_pro_dis
        num_pro_dis=self.num_pro_dis
        dependencies=self.dependencies
        trained_models=self.trained_models
        df=self.df
        sample_indices=self.sample_indices
        walk_graph=self.walk_graph
        KDE=self.KDE
        N = self.N
        T = self.T
        num_pre_dis=self.num_pre_dis
        trees=self.trees

        
        selected_sample = df.iloc[sample_indices[i]].to_dict()
        layers=self.layers
        node = random.choice(layers[0]) 
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
                if KDE:
                    CN = N
                    max_count = 5
                    c_count = 0
                    while c_count < max_count:
                        result = estimate_conditional_distribution_KDE(
                            df, num_key, str_key,
                            {key: selected_sample[key] for key in conditional_keys},
                            num_pre_dis, node, N=CN, trees=trees,
                            method='kde', bw_method=bw_method                                  
                        )
                        if result:
                            break
                        else:
                            CN *= 2
                            c_count += 1
                    if result:
                        selected_sample[node] = sample_from_dict(result, top_p=None, temperature=T)
                else:
                    if node in trained_models:
                        
                        inputs = {}
                        for conditional_key in conditional_keys:
                            inputs[conditional_key]=selected_sample[conditional_key]
                        model_wrapper = trained_models[node]
                        result = predict(node, inputs, dependencies,  num_pre_dis[node], model_wrapper, num_key, str_key, self.scalers, self.encoders)
    
                        if result is not None:
                            selected_sample[node] = result
                        else:
                            raise Exception("No result.")
                    else:
                        CN = N
                        max_count = 5
                        c_count = 0
                        while c_count < max_count:
                            result = estimate_conditional_distribution_KDE(
                                df, num_key, str_key,
                                {key: selected_sample[key] for key in conditional_keys},
                                num_pre_dis, node, N=CN, trees=trees,
                                method='kde', bw_method=bw_method                                  
                            )
                            if result:
                                break
                            else:
                                CN *= 2
                                c_count += 1
                        if result:
                            selected_sample[node] = sample_from_dict(result, top_p=None, temperature=T)
        
                nodes_next_turn += walk_graph[node]
            nodes = nodes_next_turn
        return selected_sample
        
    def sample(self, sample_num=100, method='nf'):
        SAMPLE_NUMS = sample_num
        self.N = 1  
        self.T = 1

        KDE=(method=='KDE') 
        self.KDE=KDE
        bw_method=self.bw_method
        num_key=self.num_key
        str_key=self.str_key
        str_pro_dis=self.str_pro_dis
        dependencies=self.dependencies
        trained_models=self.trained_models
        df=self.df
        
        if SAMPLE_NUMS<=len(df):
            epoch_num=1
        else:
            epoch_num=SAMPLE_NUMS//len(df)+1
            SAMPLE_NUMS=len(df)-1
        
        print("Start Generating..")
            
        generated_samples=[]
        
        for epoch in range(epoch_num):
            print(f"Epoch: {epoch}/{epoch_num}")
            sample_indices = random.sample(range(len(df)), SAMPLE_NUMS)
            self.sample_indices=sample_indices
            with ThreadPoolExecutor(max_workers=32) as executor:
                generated_samples += list(tqdm(executor.map(self.generate_sample, range(SAMPLE_NUMS)), total=SAMPLE_NUMS))
        return generated_samples

    def fill_the_blank(self, df_input, target_feature,  KDE=True, bw_method='scott', T=1.0, N=1):
    
        generated = []
        trees=self.trees
        model_dict=self.trained_models 
        dependencies=self.dependencies
        num_key=self.num_key
        str_key=self.str_key
        num_pre_dis=self.num_pre_dis

        df_input=pd.read_csv(df_input)
        
        for _, row in df_input.iterrows():
            known_values = row.drop(labels=[target_feature]).to_dict()
            conditional_keys = dependencies.get(target_feature, [])
            available_conditions = {k: known_values[k] for k in conditional_keys if k in known_values}
    
            result = None
    
            # 尝试使用模型预测
            if target_feature in model_dict:
                model_wrapper = model_dict[target_feature]
                result = predict(target_feature, available_conditions, dependencies,
                                     num_pre_dis[target_feature], model_wrapper,
                                     num_key, str_key, scalers=self.scalers, encoders=self.encoders)
    
            # fallback: KDE
            if result is None and KDE:
                CN = N
                max_count = 5
                c_count = 0
                while c_count < max_count:
                    result_dict = estimate_conditional_distribution_KDE(
                        df_input, num_key, str_key,
                        available_conditions,
                        num_pre_dis, target_feature, N=CN, trees=trees,
                        method='kde', bw_method=bw_method
                    )
                    if result_dict:
                        result = sample_from_dict(result_dict, top_p=None, temperature=T)
                        break
                    else:
                        CN *= 2
                        c_count += 1
    
            generated.append(result)
    
        df_output = df_input.copy()
        df_output["generated_" + target_feature] = generated
        return df_output

if __name__=="__main__":
    spada=SPADA()
    spada.fit(data='', 
          description="None",
          #response=None,
          target_feature='')
    df=spada.fill_the_blank('datasets/iris/test.csv', 'SepalLengthCm', "output.csv")
    df.to_csv("output.csv", index=False, encoding='utf-8')