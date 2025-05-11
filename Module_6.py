import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler



#Reads  in the csv file
df = pd.read_csv("/Users/devinreed/Downloads/INST414/Module6/tmdb_5000_movies.csv")

#Cleans data and removes movies with null values and 0 budget or revenue
df = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()

#success label
df['success'] = (df['revenue'] > 1.5 * df['budget']).astype(int)



def get_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return [g['name'] for g in genres if 'name' in g]
    except:
        return []
    

df['genre_list'] = df['genres'].apply(get_genres)

#top 5 most common genres 
all_genres = [genre for sublist in df['genre_list'] for genre in sublist]
top_genre = pd.Series(all_genres).value_counts().nlargest(5).index.tolist()



for genre in top_genre:
    df[f'genre_{genre}'] = df['genre_list'].apply(lambda x: int(genre in x))


#converts release date to date time format
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_month'] = df['release_date'].dt.month.fillna(0).astype(int)


features = ['budget', 'runtime', 'release_month'] + [f'genre_{g}' for g in top_genre] + ['success']

df_model = df[features].dropna()
x = df_model.drop(columns = 'success')
y = df_model['success']




#train and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=40)

scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)


#random forest classification
rf = RandomForestClassifier(n_estimators=100, random_state=40)
rf.fit(X_train_scale, y_train)

y_pred = rf.predict(X_test_scale)

#prints performance
print("Classification Report: \n", classification_report(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))




test_df = X_test.copy()
test_df['true'] = y_test
test_df['pred'] = y_pred
test_df['correct'] = test_df['true'] == test_df['pred']
test_df['title'] = df.loc[X_test.index, 'title']



misclassified = test_df[test_df['correct'] == False]
sampled_errors = misclassified.sample(5, random_state = 40)
print(sampled_errors[['title', 'budget', 'runtime', 'release_month'] + [col for col in sampled_errors.columns if 'genre_' in col] + ['true', 'pred']])

