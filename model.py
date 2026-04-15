import ast
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_director(row):
    for credit in row["crew"]:
        if credit["job"] == "Director":
            return credit["name"]
    return ""


def get_attrs(row, attr):
    return [mem["name"] for mem in row[attr]]


def build_input(row):
    overview = row.get("overview", "") or ""
    genres = " ".join(row.get("genre_names", []))
    title = row.get("original_title", "") or ""
    keywords = " ".join(row.get("keyword_names", []))
    return f"{overview} {genres} {keywords} {title}".strip()

def genre_classifier(df):
    pass


if __name__ == "__main__":
    credits_df = pd.read_csv("data/tmdb_5000_credits.csv")
    movies_df = pd.read_csv("data/tmdb_5000_movies.csv")

    df = pd.merge(movies_df, credits_df, left_on="id", right_on="movie_id")
    df = df.drop(columns=["movie_id"])
    df["overview"] = df["overview"].fillna("").astype(str)

    df["crew"] = df["crew"].apply(ast.literal_eval)
    df["director"] = df.apply(get_director, axis=1)
    df = df[df["director"] != ""]

    # strict rule: only filter out directors with fewer than 4 movies
    dir_counts = df["director"].value_counts()
    valid_dirs = dir_counts[dir_counts >= 5].index
    df = df[df["director"].isin(valid_dirs)].copy()

    df["cast"] = df["cast"].apply(ast.literal_eval)
    df["cast_names"] = df.apply(get_attrs, args=("cast",), axis=1)

    df["genres"] = df["genres"].apply(ast.literal_eval)
    df["genre_names"] = df.apply(get_attrs, args=("genres",), axis=1)

    df["keywords"] = df["keywords"].apply(ast.literal_eval)
    df["keyword_names"] = df.apply(get_attrs, args=("keywords",), axis=1)

    df["input"] = df.apply(build_input, axis=1)
    df = df.reset_index(drop=True)

    train, test = train_test_split(
        df,
        test_size=0.15,
        random_state=100,
        stratify=df["director"],
    )

    train_text = train["input"].tolist()
    test_text = test["input"].tolist()

    word_tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        stop_words="english",
    )
    X_train_word = word_tfidf.fit_transform(train_text)
    X_test_word = word_tfidf.transform(test_text)

    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=30000,
        min_df=2,
        sublinear_tf=True,
    )
    X_train_char = char_tfidf.fit_transform(train_text)
    X_test_char = char_tfidf.transform(test_text)

    train_emb = sentence_model.encode(
        train_text,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    test_emb = sentence_model.encode(
        test_text,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # combine features
    X_train = hstack([X_train_word, X_train_char, csr_matrix(train_emb),]).tocsr()
    X_test = hstack([X_test_word, X_test_char, csr_matrix(test_emb),]).tocsr()

    y_train = train["director"]
    y_test = test["director"]

    clf = LinearSVC(C=1.5, class_weight="balanced", max_iter=5000)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=100)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring="accuracy")
    print(f"CV accuracy on train: {cv_scores.mean():.3f} (std: {cv_scores.std():.3f})")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Holdout accuracy: {accuracy_score(y_test, y_pred):.3f}")

    # calibrate
    calibrated = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)

    with open("model/director_classifier.sav", "wb") as f:
        pickle.dump(calibrated, f)

    with open("model/word_tfidf.sav", "wb") as f:
        pickle.dump(word_tfidf, f)

    with open("model/char_tfidf.sav", "wb") as f:
        pickle.dump(char_tfidf, f)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train.to_pickle("data/train_df.pkl")
    test.to_pickle("data/test_df.pkl")
    np.save("model/train_embeddings.npy", train_emb)
    np.save("model/test_embeddings.npy", test_emb)