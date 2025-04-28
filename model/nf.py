from sklearn.preprocessing import StandardScaler, LabelEncoder
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import AffineCouplingTransform, ReversePermutation, CompositeTransform
import torch
import torch.nn as nn
from collections import namedtuple
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

class SwiGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(in_features, out_features)
        self.silu = nn.SiLU()  

    def forward(self, x):
        return self.silu(self.linear1(x)) * self.linear2(x) 
        
class ContextNet(nn.Module):
    def __init__(self, in_features, out_features, context_features, dim=8, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features + context_features, 2*dim),
            nn.LayerNorm(2*dim),
            SwiGLU(2*dim, 2*dim),
            nn.Dropout(dropout_rate),
            nn.Linear(2*dim, dim),
            nn.LayerNorm(dim),
            nn.Linear(dim, out_features)
        )

    def forward(self, x, context=None):
        if context is not None:
            x = torch.cat([x, context], dim=-1)
        return self.net(x)

def nearest_valid_value(value, range_dict):
    min_val = range_dict['min']
    max_val = range_dict['max']
    precision = range_dict['precision']
    
    nearest_value = min_val + round((value - min_val) / precision) * precision
    
    return max(min_val, min(nearest_value, max_val))


def predict(feature_name, input_dict, dependencies, value_range, trained_models, num_key, str_key, scalers, encoders):

    if feature_name not in trained_models:
        raise ValueError(f"Feature {feature_name} holds no dependency！")

    model_wrapper = trained_models[feature_name]
    model_type = model_wrapper.type
    model = model_wrapper.model

    feature_dependencies = dependencies[feature_name]
    input_data = {}

    for dep in feature_dependencies:
        if dep in num_key:
            input_data[dep] = input_dict[dep]
        elif dep in str_key:
            enc = encoders[dep]
            input_data[dep] = enc.transform([input_dict[dep]])[0]

    input_df = pd.DataFrame([input_data])

    for dep in feature_dependencies:
        if dep in num_key:
            input_df[dep] = scalers[dep].transform(input_df[[dep]])

    if model_type == "gaussian":
        predicted_value = model.predict(input_df.values)[0]
    elif model_type == "flow":
        input_tensor = torch.tensor(input_df.values, dtype=torch.float32)
        predicted_value = model.sample(1, context=input_tensor).item()
    else:
        raise ValueError(f"Unknown model type：{model_type}")

    predicted_value = scalers[feature_name].inverse_transform(
        np.array([[predicted_value]])
    )[0][0]

    return nearest_valid_value(predicted_value, value_range)