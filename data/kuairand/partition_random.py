import pandas as pd


# The "nrows=10000" argument reads the first 10000 lines of each file

df_rand = pd.read_csv("./random.csv",header = None,sep = ' ')
df_rand = df_rand.sample(frac=1.0) #shuffle dataset

rows,cols = df_rand.shape
val_num = int(rows*0.25)

df_val = df_rand.iloc[0:val_num,:]
df_test = df_rand.iloc[val_num:rows,:]

df_val.to_csv("val.csv",header = None,index = None,sep = ' ')
df_test.to_csv("test.csv",header = None,index = None,sep = ' ')

