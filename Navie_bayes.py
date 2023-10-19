import pandas as pd
df = pd.read_csv("C://Users//vamsi//Downloads//dataset.csv")
df_dt = df.copy()
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.calibration import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df_dt = df_dt.drop('track_id', axis=1)
df_dt = df_dt.drop(['album_name','track_name'], axis=1)
label_encoder = LabelEncoder()
df_dt['artist_encoded'] = df_dt['artists'].str.split(',').str[0].str.strip()
df_dt['artist_encoded'] = label_encoder.fit_transform(df_dt['artist_encoded'])
X= df_dt.drop(['track_genre', 'artists'], axis=1)
y = df_dt['track_genre']
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size = 0.1, random_state = 0)
classifier = GaussianNB()
classifier.fit(X_Train, Y_Train)
y_pred = classifier.predict(X_Test)
accuracy = accuracy_score(Y_Test, y_pred)
confusion_mat = confusion_matrix(Y_Test, y_pred)
classification_rep = classification_report(Y_Test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion_mat}")
print(f"Classification Report:\n{classification_rep}")

file_path = ("C://Users//vamsi//Downloads//"
             ""
             "classification_report.txt")

# Open the file in write mode and write the classification report to it
with open(file_path, "w") as file:
    file.write(classification_rep )

# The classification report is now saved to the specified file
print(f"Classification report saved to {file_path}")
