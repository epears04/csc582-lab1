import sys
import ast
import pandas as pd

def main():
    # is input text optional?
    if len(sys.argv) != 2: 
        print("Usage: robotproducer.py [input.txt]")
        with open(sys.argv[1], "r") as f:
            text = f.read()

def get_director(row):
    for credit in row['crew']:
        if credit['job'] == 'Director':
            return credit['name']
    return None

def get_cast(row):
    cast_mems = []
    for mem in row['cast']:
        cast_mems.append(mem['name'])
    return cast_mems
    

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
    df['cast_names'] = df.apply(get_cast, axis=1)
    print(df.at[0,'cast_names'])

    print(df.head())
    print(df.columns)

    # main()