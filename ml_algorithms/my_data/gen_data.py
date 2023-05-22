import pandas as pd
import random
import numpy as np
import string


def make_dataset(num_samples, num_features):
    df = pd.DataFrame()
    # types = ["float", "int", "string"]
    types = ["float", "int"]
    for i in range(num_features):
        data_type = random.choice(types)
        if data_type == "int":
            df['feature_' + str(i)] = random.sample(range(0, 9999999), num_samples)
        elif data_type == "float":
            df['feature_' + str(i)] = np.random.uniform(low=-10, high=10, size=(num_samples,)).tolist()
        elif data_type == "string":
            string_list = []
            for j in range(num_samples):
                string_list.append(''.join(random.choice(string.ascii_letters + string.digits + string.punctuation)
                                           for j in range(5)))
            df['feature_' + str(i)] = string_list
    df['output'] = np.random.uniform(low=-10, high=10, size=(num_samples,)).tolist()
    return df


my_data = make_dataset(100, 100)
my_data.to_csv('my_data_2.csv', index=False)
print(my_data)
