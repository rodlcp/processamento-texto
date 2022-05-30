import pandas as pd
import os

original_data_path = 'original_data'

files = os.listdir(original_data_path)

data = pd.concat([pd.read_csv(os.path.join(original_data_path, file), on_bad_lines='skip') for file in files])
data.drop_duplicates(inplace = True)
data.reset_index(drop=True, inplace = True)

str_fields = ['Title', 'Source title', 'Abstract', 'Author Keywords', 'Index Keywords', 'Document Type']
int_fields = ['Year', 'Cited by']

data = data[str_fields + int_fields]

for i in str_fields:
    data[i] = data[i].fillna('').astype(str)

for i in int_fields:
    data[i] = data[i].fillna(0).astype(int)

data.to_parquet('concatenated_data.parquet', index = False)