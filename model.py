import ast
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_director(row):
    for credit in row['crew']:
        if credit['job'] == 'Director':
            return credit['name']
    return ""

def get_attrs(row, attr):
    results = []
    for mem in row[f'{attr}']:
        results.append(mem['name'])
    return results

def classifier(train_df, test_df, train_embeddings, test_embeddings):
    y_train = train_df['director']
    y_test = test_df['director']

    clf = LinearSVC(max_iter=3000, class_weight='balanced', C=0.1)
    clf.fit(train_embeddings, y_train)
    y_pred = clf.predict(test_embeddings)

    print(f"Accuracy: {(sum(y_pred == y_test)) / len(y_test):.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    pickle.dump(clf, open('model/director_classifier.sav', 'wb'))

if __name__ == "__main__":
    credits_df = pd.read_csv("data/tmdb_5000_credits.csv")
    movies_df = pd.read_csv("data/tmdb_5000_movies.csv")
    df = pd.merge(movies_df, credits_df, left_on="id", right_on="movie_id")
    df = df.drop(columns=['movie_id'])
    df['overview'] = df['overview'].fillna("").astype(str)

    # get director
    df['crew'] = df['crew'].apply(ast.literal_eval)
    df['director'] = df.apply(get_director, axis=1) 
    df = df[df['director'] != ""]

    # drop if director appears less than 4 times
    dir_counts = df['director'].value_counts()
    valid_dirs = dir_counts[dir_counts >= 8].index
    df = df[df['director'].isin(valid_dirs)]

    # get cast member namesp
    df['cast'] = df['cast'].apply(ast.literal_eval)
    df['cast_names'] = df.apply(get_attrs, args=("cast",), axis=1)

    #get genre
    df['genres'] = df['genres'].apply(ast.literal_eval)
    df['genre_names'] = df.apply(get_attrs, args=("genres",), axis=1)
    
    print("Size of origninal df:", df.size)

    train, test = train_test_split(df, test_size=0.1, random_state=100, shuffle=True, stratify=df['director'])

    print("Size of train df:", train.size)
    print("Size of test df:", test.size)

    train_overviews = train['overview'].tolist()
    train_embeddings = model.encode(train_overviews)

    test_overviews = test['overview'].tolist()
    test_embeddings = model.encode(test_overviews)

    # save
    np.save('model/train_embeddings.npy', train_embeddings)
    np.save('model/test_embeddings.npy', test_embeddings)
    train.to_pickle('data/train_df.pkl')
    test.to_pickle('data/test_df.pkl')

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    classifier(train, test, train_embeddings, test_embeddings)
