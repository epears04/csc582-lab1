import sys
import pickle 
import random
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = np.load('model/train_embeddings.npy')

with open('data/train_df.pkl', 'rb') as f:
    df = pickle.load(f)

with open('model/cross_classifier.sav', 'rb') as f:
    clf = pickle.load(f)

def evaluate():
    with open('data/test_df.pkl', 'rb') as f:
        test_df = pickle.load(f)

    correct = 0
    total_score = 0
    for i in range(len(test_df)):
        overview = test_df.iloc[i]['overview']
        sim_idx, sim = get_similar(model.encode(overview))
        embedding_predicted_dir= predict_director(sim_idx, sim)
        ml_predicted_dir = predict_director_ml(model.encode(overview).reshape(1, -1))
        actual_dir = test_df.iloc[i]['director']
        actual_cast = test_df.iloc[i]['cast_names']
        predicted_cast = predict_cast(sim_idx, sim)

        score = 0
        correct_dir = False

        if embedding_predicted_dir == actual_dir:
            score += 20
            correct_dir = True
            correct += 1
            print("! from embedding")
        if ml_predicted_dir == actual_dir:
            if not correct_dir:
                score += 20
                correct += 1
            print("! from ml")

        base_hits = 0
        bonus_hits = 0

        actual_cast_set = set(actual_cast)
        actual_top5 = set(actual_cast[:5])

        for actor in predicted_cast:
            if actor in actual_cast_set and base_hits < 5:
                base_hits += 1
            if actor in actual_top5:
                bonus_hits += 1

        score += 10 * min(base_hits, 5)
        score += 5 * min(bonus_hits, 5)
        print(f"{test_df.iloc[i]['original_title']}: {score}")
        total_score += score

    print(f"average score = {total_score / len(test_df):.2f}")
    print(f'director accuracy = {(correct / len(test_df)) * 100:.2f}%')
    
def get_similar(embedding, k=100):
    query = embedding.reshape(1, -1)
    similarities = model.similarity(query, embeddings).flatten().numpy()
    top_k_indices = np.argsort(similarities)[::-1][:k]
    top_k_sims = similarities[top_k_indices]
    return top_k_indices, top_k_sims

def predict_director(top_k, top_k_sim):
    director_scores = {}
    for idx, score in zip(top_k, top_k_sim):
        director = df.iloc[idx]['director']
        s = float(score) ** 2
        if director:
            director_scores[director] = director_scores.get(director, 0) + s
    
    predicted = max(director_scores, key=director_scores.get)
    return predicted

def predict_director_ml(embedding):
    return clf.predict(embedding)

def predict_cast(top_k, top_k_sim):
    cast_scores = {}
    for idx, score in zip(top_k, top_k_sim):
        cast_members = df.iloc[idx]['cast_names']
        for mem in cast_members:
           cast_scores[mem] = cast_scores.get(mem, 0) + float(score) ** 2

    return sorted(cast_scores, key=cast_scores.get, reverse=True)[:20]

def predict_movie(text):
    embedding = model.encode(text)

    top_k_idx, top_k_sim = get_similar(embedding, embeddings, k=50)

    predicted_dir, director_scores = predict_director_hybrid(
        embedding, top_k_idx, top_k_sim, clf
    )

    predicted_cast = predict_cast(
        top_k_idx, top_k_sim, predicted_dir, top_n=10
    )

    predicted_title = suggest_title(text, top_k_idx)

    return predicted_title, predicted_dir, predicted_cast

def main():
    if len(sys.argv) > 2: 
        print("Usage: robotproducer.py [input.txt]")
    elif len(sys.argv) == 1:
        # run testing data
        evaluate()
    else:
        # run normally
        with open(sys.argv[1], "r") as f:
            text = f.read()
        predict_movie(text)

if __name__ == "__main__":
    main()