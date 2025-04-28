import pandas as pd
import random
from lookup import filter_rows_by_column_for_str_features, filter_rows_by_column_for_num_features
from model.gmm import estimate_gmm_probabilities, estimate_initiate_pro_distribution_for_num_features
import numpy as np
#from numba import jit


def build_distribution_dict_for_str_features(df, str_keys, verbose=False):
    if type(df) is str:
        df=pd.read_csv(df)
    str_pro_dis={key:{} for key in str_keys}
    for key in str_keys:
        count=0
        for value in df[key]:
            if value not in str_pro_dis[key]:
                str_pro_dis[key][value]=1
            else:
                str_pro_dis[key][value]+=1
            count+=1
        for value in str_pro_dis[key]:
            str_pro_dis[key][value]/=count
    if verbose:
        for key in str_pro_dis:
            if not str_pro_dis[key]:
                print(f'Warning: the feature {key} has no distribution.')
    return str_pro_dis


def build_precision_dict_for_num_features(df, num_keys, verbose=False):

    if type(df) is str:
        df=pd.read_csv(df)
    num_pre_dis={key:{'min':0,'max':0, 'precision':0} for key in num_keys} #[min, max, precision]
    for key in num_keys:
        l=list(set([float(i) for i in df[key]])) 
        l.sort()
        min_diff=float("inf")
        for i in range(len(l)-1):
            min_diff=min(l[i+1]-l[i], min_diff)
        num_pre_dis[key]['precision']=min_diff
        num_pre_dis[key]['min']=min(l)
        num_pre_dis[key]['max']=max(l)
    if verbose:
        for key in num_pre_dis:
            if num_pre_dis[key]['precision']==0 or num_pre_dis[key]['precision']==float("inf"):
                print(f"Warning: the feature {key} may overflow.")
    return num_pre_dis



def pro_distribution_in_conditions(df, feature, conditions=None, num_keys=None, str_keys=None, \
                                   num_pre_dis=None, fuzzy_match=True, fuzzy_size=500, verbose=False,\
                                   return_gmm_probabilities=False, n_components=3):

    assert num_keys or str_keys
    if num_keys:
        assert num_pre_dis
        assert len(df)>=fuzzy_size
        
    if type(df) is str:
        df=pd.read_csv(df)
    tmp_df=df.copy()
    for key in conditions:
        if verbose and key not in num_keys and key not in str_keys:
            print(f'Warning: the feature {key} is neither a number nor a string.')
        value=conditions[key]
        if value is None and verbose:
            print(f'Warning: The feature value {value} is missing.')
        if str_keys:
            if key in str_keys:
                tmp_df=filter_rows_by_column_for_str_features(tmp_df, key, value)
        if num_keys:
            if key in num_keys:
                if fuzzy_match:
                    tmp_df=filter_rows_by_column_for_num_features(tmp_df, key, value, num_pre_dis[key]['precision'], fuzzy_size=fuzzy_size)
                else:
                    tmp_df=filter_rows_by_column_for_num_features(tmp_df, key, value)
        if tmp_df is None:
            return False

    if feature in num_keys and return_gmm_probabilities and int(n_components)>0:
        return estimate_gmm_probabilities([float(value) for value in tmp_df[feature]], num_pre_dis[feature], n_components=int(n_components))
        
    feature_dis={}
    count=0
    for value in tmp_df[feature]:
        if value not in feature_dis:
            feature_dis[value]=1
        else:
            feature_dis[value]+=1
        count+=1
    for value in feature_dis:
        feature_dis[value]/=count
    return feature_dis


def sample_from_dict(prob_dict, top_p=None, temperature=1.0):
    keys, probabilities = zip(*prob_dict.items())  
    probabilities = np.array(probabilities, dtype=np.float64)

    if temperature != 1.0:
        probabilities = np.power(probabilities+ 1e-10, 1.0 / temperature)
    
    probabilities /= probabilities.sum()

    if top_p is not None:
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_keys = np.array(keys)[sorted_indices]
        sorted_probs = probabilities[sorted_indices]

        cumulative_probs = np.cumsum(sorted_probs)
        cutoff_idx = np.searchsorted(cumulative_probs, top_p, side="right")

        filtered_keys = sorted_keys[:cutoff_idx + 1]
        filtered_probs = sorted_probs[:cutoff_idx + 1]

        filtered_probs /= filtered_probs.sum()

        return np.random.choice(filtered_keys, p=filtered_probs)
    else:
        return np.random.choice(keys, p=probabilities)


