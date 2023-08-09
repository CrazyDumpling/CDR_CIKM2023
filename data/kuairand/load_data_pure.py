import pandas as pd


# The "nrows=10000" argument reads the first 10000 lines of each file

df_rand = pd.read_csv("data/log_random_4_22_to_5_08_pure.csv")

df1 = pd.read_csv("data/log_standard_4_08_to_4_21_pure.csv", nrows=10000)
df2 = pd.read_csv("data/log_standard_4_22_to_5_08_pure.csv")

df_rand = df_rand.iloc[:,[0,1,5]]
df2 = df2.iloc[:,[0,1,5]]
df_rand.to_csv("random.csv",header = None,index = None,sep = ' ')
df2.to_csv("user.csv",header = None,index = None,sep = ' ')

df_rand[df_rand < 1] = -1
df_rand.to_csv("random1.csv",header = None,index = None,sep = ' ')
print(len(df_rand))


user_features = pd.read_csv("data/user_features_pure.csv")

video_features_basic = pd.read_csv("data/video_features_basic_pure.csv", nrows=10000)
video_features_statistics = pd.read_csv("data/video_features_statistic_pure.csv", nrows=10000)
