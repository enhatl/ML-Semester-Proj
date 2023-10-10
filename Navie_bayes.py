import pandas as pd

df = pd.read_csv("C://Users//vamsi//Downloads//dataset.csv")
df_dt = df.copy()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.calibration import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df_dt = df_dt.drop('track_id', axis=1)
#album name and track name are not going to be useful either
df_dt = df_dt.drop(['album_name','track_name'], axis=1)
label_encoder = LabelEncoder()

#We are assuming the first artist is the main artist and will give the most information.
df_dt['artist_encoded'] = df_dt['artists'].str.split(',').str[0].str.strip()  #Extract first artist
df_dt['artist_encoded'] = label_encoder.fit_transform(df_dt['artist_encoded'])  #Apply label encoding

X= df_dt.drop(['track_genre', 'artists'], axis=1) #dont need artists column anymore
y = df_dt['track_genre']

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size = 0.1, random_state = 0)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_Train, Y_Train)

y_pred = classifier.predict(X_Test)

accuracy = accuracy_score(Y_Test, y_pred)
confusion_mat = confusion_matrix(Y_Test, y_pred)
classification_rep = classification_report(Y_Test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion_mat}")
print(f"Classification Report:\n{classification_rep}")
