import re
import pandas as pd

def is_number_regex(s):
    if s:
        pattern = r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$'  
        return bool(re.match(pattern, s))
    else:
        return False


def number_type_key(dataset_pth, verbose=False, blank=None):
    if type(dataset_pth) is str:
        df=pd.read_csv(dataset_pth)
    else:
        df=dataset_pth
    key_value_dict={key:[value for value in df[key]] for key in df}
    
    num_key=[key for key in key_value_dict]
    str_key=list()
    
    for key in key_value_dict:
        for value in key_value_dict[key]:
            if not is_number_regex(str(value)) and str(value)!=blank:
                str_key.append(key)
                num_key.remove(key)
                break
                
    if verbose:
        print(f"The following features hold strings:{str_key}")
        print(f"The following features hold numbers:{num_key}")
    return num_key, str_key