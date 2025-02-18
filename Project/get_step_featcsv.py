import os 
import pandas as pd 

df = pd.read_csv('RESULTS_DIRECTORY/process_list_autogen.csv') 
ids1 = [i[:-4] for i in df.slide_id]
ids2 = [i[:-3] for i in os.listdir('RESULTS_DIRECTORY/patches')]
df['slide_id'] = ids1
ids = df['slide_id'].isin(ids2)
sum(ids)
df.loc[ids].to_csv('RESULTS_DIRECTORY/Step_2.csv',index=False)
