import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
from numba import jit
from copulas.univariate import GaussianKDE
from copulas.bivariate import Clayton, Gumbel, Frank
from sklearn.neighbors import NearestNeighbors

def safe_gmm(values, n_components=3):
    values = np.array(values).reshape(-1, 1)
    
    if len(values) < n_components:
        return None  
    
    try:
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(values)
        
        return gmm
    except Exception as e:
        print(f"safe_gmm error: {e}")
        return None  



def safe_kde(values, bw_method='scott'):
    values = np.array(values, dtype=float)

    if len(values) < 3:
        return None  

    values += np.random.normal(0, 1e-3, size=len(values))

    try:
        kde = gaussian_kde(values, bw_method=bw_method)
        return kde
    except np.linalg.LinAlgError:
        return None 

def adaptive_kde(values, N_neighbors=10):
    values = np.array(values, dtype=float).reshape(-1, 1)
    
    if len(values) < 3:
        return None  
    
    nbrs = NearestNeighbors(n_neighbors=N_neighbors, algorithm='ball_tree').fit(values)
    distances, _ = nbrs.kneighbors(values)
    
    bandwidths = np.mean(distances[:, -1])  
    
    try:
        kde = gaussian_kde(values.T, bw_method=bandwidths) 
        return kde
    except Exception as e:
        print(f"safe_kde error: {e}")
        return None  

def safe_copula(values, copula_type='gaussian'):
    values = np.array(values, dtype=float)

    if len(values) < 3:
        return None  

    values += np.random.normal(0, 1e-3, size=len(values))  

    try:
        kde = GaussianKDE()
        kde.fit(values)
        transformed_values = kde.cdf(values)  

        if copula_type == 'clayton':
            copula = Clayton()
        elif copula_type == 'gumbel':
            copula = Gumbel()
        elif copula_type == 'frank':
            copula = Frank()
        else:
            raise ValueError(f"Unknown copula type: {copula_type}")
        
        copula.fit(transformed_values.reshape(-1, 1)) 
        
        return lambda x: copula.probability_density(kde.cdf(np.array(x)).reshape(-1, 1)) 

    except Exception as e:
        print(f"safe_copula 失败: {e}")
        return None 

@jit()
def uni(probabilities, discrete_values, precision):
    probabilities /= probabilities.sum()
    return {round(val, int(-np.log10(precision))): prob for val, prob in zip(discrete_values, probabilities)}

    
def estimate_conditional_distribution_KDE(df, num_key, str_key, conditions, num_pre_dis, target_key, \
                                          N=5, trees=None, method='kde', bw_method="scott"):
    num_df_values = {key: df[key].to_numpy() for key in num_key}
    str_df_values = {key: df[key].astype(str).str.strip().to_numpy() for key in str_key}
    

    condition_mask = np.ones(df.shape[0], dtype=bool)
    
    for key, value in conditions.items():
        if key in num_key:
            df_values = num_df_values[key] 
            precision = num_pre_dis[key]['precision']
            value = float(value)
            if trees:
                dist, ind = trees[key].query([[value]], k=50)
                condition_mask &= np.isin(np.arange(len(df)), ind[0])
            else:
                condition_mask &= np.abs(df_values - value) <= N * precision
    
        elif key in str_key:
            value = str(value).strip()
    
            condition_mask &= str_df_values[key] == value
    
        else:
            print(f"Error: {key} is neither a number nor string.")
            return False
    
    filtered_df = df[condition_mask]
    if len(filtered_df) == 0:
        return {}
        
    if target_key in str_key:  
        return filtered_df[target_key].value_counts(normalize=True).to_dict()
    
    elif target_key in num_key:  
        value_range = num_pre_dis[target_key]
        min_val, max_val, precision = value_range['min'], value_range['max'], value_range['precision']

        values = filtered_df[target_key].dropna().to_numpy() 
        if len(values) == 0: 
            return {}

        if method=='kde':
            kde = safe_kde(values, bw_method=bw_method)
        elif method=='adaptive_kde':
            kde = adaptive_kde(values)
        elif method=='gmm':
            kde = safe_gmm(values)
        else:
            raise ValueError(f"Unknown method type: {method}")
        
        if kde is None:
            return None
            
        num_samples = min(500, int((max_val - min_val) / precision))

        discrete_values = np.arange(min_val, max_val + precision, precision)
        
        probabilities = kde(discrete_values)

        return uni(probabilities, discrete_values, precision)

    else:
        raise ValueError(f"'{target_key}' is not in num_key nor str_key!")


def estimate_gmm_probabilities(samples, params, n_components=3):
    min_val, max_val, precision = params['min'], params['max'], params['precision']
    
    samples = np.array(samples).reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(samples)
    
    x_values = np.arange(min_val, max_val + precision, precision)
    x_values = x_values[x_values <= max_val].reshape(-1, 1)  
    
    densities = np.exp(gmm.score_samples(x_values))
    probabilities = densities / densities.sum()
    
    return dict(zip(x_values.flatten(), probabilities))


def estimate_initiate_pro_distribution_for_num_features(df, num_keys, num_pre_dis, verbose=False, n_components=3): 
    
    if type(df) is str:
        df=pd.read_csv(df)
    num_pro_dis={key:None for key in num_keys} 
    for key in num_keys:
        num_pro_dis[key]=estimate_gmm_probabilities([float(value) for value in df[key]], num_pre_dis[key], n_components=n_components)
    return num_pro_dis