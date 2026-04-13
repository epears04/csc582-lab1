import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

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

# def get_embeddings(row):
#     overview = row['overview']
#     if pd.isna(overview):
#         overview = ""
#     return model.encode(str(overview))

if __name__ == "__main__":
    credits_df = pd.read_csv("data/tmdb_5000_credits.csv")
    movies_df = pd.read_csv("data/tmdb_5000_movies.csv")
    df = pd.merge(movies_df, credits_df, left_on="id", right_on="movie_id")
    df = df.drop(columns=['movie_id'])
    df['overview'] = df['overview'].fillna("").astype(str)

    # get director
    df['crew'] = df['crew'].apply(ast.literal_eval)
    df['director'] = df.apply(get_director, axis=1) 

    # get cast member names
    df['cast'] = df['cast'].apply(ast.literal_eval)
    df['cast_names'] = df.apply(get_attrs, args=("cast",), axis=1)

    #get genre
    df['genres'] = df['genres'].apply(ast.literal_eval)
    df['genre_names'] = df.apply(get_attrs, args=("genres",), axis=1)
    
    print("Size of origninal df:", df.size)

    train, test = train_test_split(df, test_size=0.1, random_state=100, shuffle=True)

    print("Size of train df:", train.size)
    print("Size of test df:", test.size)

    train_overviews = train['overview'].tolist()
    train_embeddings = model.encode(train_overviews)

    # save
    np.save('data/train_embeddings.npy', train_embeddings)
    train.to_pickle('data/train_df.pkl')
    test.to_pickle('data/test_df.pkl')
