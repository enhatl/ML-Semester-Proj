# https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/tree/main
import pandas as pd
url = 'https://raw.githubusercontent.com/enhatl/ML-Semester-Proj/main/dataset.csv'
df = pd.read_csv(url,index_col=0,parse_dates=[0])

