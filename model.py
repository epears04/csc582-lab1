import ast
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
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

def build_input(row):
    overview = row.get('overview', '') or ''
    genres = ' '.join(row.get('genre_names', []))
    title = row.get('original_title', '') or ''
    keywords = ' '.join(row.get('keyword_names', []))
    return f"{overview} {genres} {title} {keywords}".strip()

if __name__ == "__main__":
    # combine datasets 
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

    # get keywords
    df['keywords'] = df['keywords'].apply(ast.literal_eval)
    df['keyword_names'] = df.apply(get_attrs, args=("keywords",), axis=1)

    df['input'] = df.apply(build_input, axis=1)

    df = df.reset_index(drop=True)
    embeddings = model.encode(df['input'].tolist(), show_progress_bar=True)
    np.save('model/embeddings.npy', embeddings)

    train, test = train_test_split(df, test_size=0.1, random_state=100)
    test = test[test['director'].isin(set(train['director']))]

    lr = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.5)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
    scores = cross_val_score(lr, embeddings, df['director'], cv=skf, scoring='accuracy')
    print(f"CV accuracy: {scores.mean():.3f} (std: {scores.std():.3f})")

    lr.fit(embeddings[train.index], train['director'])
    pickle.dump(lr, open('model/cross_classifier.sav', 'wb'))

    y_test = test['director']
    y_pred = lr.predict(embeddings[test.index])

    print(classification_report(y_test, y_pred, zero_division=0))

    train.to_pickle('data/train_df.pkl')
    test.to_pickle('data/test_df.pkl')
    np.save('model/train_embeddings.npy', embeddings[train.index])
    np.save('model/test_embeddings.npy', embeddings[test.index])
