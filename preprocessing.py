import sys
import ast
import pandas as pd
from sklearn.model_selection import train_test_split

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

if __name__ == "__main__":
    credits_df = pd.read_csv("data/tmdb_5000_credits.csv")
    movies_df = pd.read_csv("data/tmdb_5000_movies.csv")
    df = pd.merge(movies_df, credits_df, left_on="id", right_on="movie_id")
    df = df.drop(columns=['movie_id'])

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
    # print(df.head())

    # for i in range(5):
    #     print("title", df.at[i, "original_title"])
    #     print(df.at[i, "genre_names"])
    #     print(df.at[i, "cast_names"])
    #     print(df.at[i, "overview"])

    train, test = train_test_split(df, test_size=0.1, random_state=100, shuffle=True)

    print("Size of train df:", train.size)
    # print(train.head())
    print("Size of test df:", test.size)
    # print(test.head())
    
    # save
    train.to_csv('data/train_df.csv')
    test.to_csv('data/test_df.csv')