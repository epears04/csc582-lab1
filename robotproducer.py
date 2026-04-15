import sys
import pickle
import re
import numpy as np
from collections import defaultdict, Counter
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
train_embeddings = np.load("model/train_embeddings.npy")

with open("data/train_df.pkl", "rb") as f:
    df = pickle.load(f)

with open("model/director_classifier.sav", "rb") as f:
    clf = pickle.load(f)

with open("model/word_tfidf.sav", "rb") as f:
    word_tfidf = pickle.load(f)

with open("model/char_tfidf.sav", "rb") as f:
    char_tfidf = pickle.load(f)

director_actor_counts = defaultdict(float)
for _, row in df.iterrows():
    director = row["director"]
    for i, actor in enumerate(row["cast_names"][:10]):
        weight = 3.0 if i < 5 else 1.0
        director_actor_counts[(director, actor)] += weight

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "his", "her", "its", "their", "our",
    "your", "my", "this", "that", "with", "from", "by", "as", "it", "he", "she",
    "they", "we", "you", "i", "not", "so", "if", "into", "up", "out", "about",
    "after", "before", "who", "which", "what", "all", "him", "them", "also", "just",
    "only", "even", "back", "over", "man", "woman", "one", "two", "time", "film",
    "movie", "story", "world", "life", "day", "new", "old", "when"
}


def build_classifier_features(overview):
    X_word = word_tfidf.transform([overview])
    X_char = char_tfidf.transform([overview])

    emb = model.encode([overview], convert_to_numpy=True, normalize_embeddings=True,)

    X = hstack([X_word, X_char, csr_matrix(emb),]).tocsr()
    return X, emb[0]

def get_similar(embedding_1d, k=50):
    similarities = model.similarity(embedding_1d, train_embeddings).flatten().numpy()
    top_k_indices = np.argsort(similarities)[::-1][:k]
    top_k_sims = similarities[top_k_indices]
    return top_k_indices, top_k_sims

def predict_director_emb(top_k, top_k_sim):
    director_scores = {}

    for idx, score in zip(top_k, top_k_sim):
        director = df.iloc[idx]["director"]
        s = float(score) ** 2
        director_scores[director] = director_scores.get(director, 0.0) + s

    predicted = max(director_scores, key=director_scores.get)
    return predicted, director_scores

def predict_director_ml(features):
    pred = clf.predict(features)[0]

    prob_dict = {}
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(features)[0]
        prob_dict = dict(zip(clf.classes_, probs))

    return pred, prob_dict

def predict_director_hybrid(features, top_k, top_k_sim, ml_weight=0.7):
    ml_pred, ml_probs = predict_director_ml(features)
    emb_pred, emb_scores = predict_director_emb(top_k, top_k_sim)

    emb_total = sum(emb_scores.values()) or 1.0
    emb_probs = {d: s / emb_total for d, s in emb_scores.items()}

    if ml_probs:
        all_directors = set(ml_probs) | set(emb_scores)
        combined = {
            d: ml_weight * ml_probs.get(d, 0.0) + (1.0 - ml_weight) * emb_probs.get(d, 0.0)
            for d in all_directors
        }
        predicted = max(combined, key=combined.get)
        return predicted, combined

    return ml_pred, {ml_pred: 1.0}

def predict_cast(top_k, top_k_sim, predicted_director="", top_n=15):
    sim_scores = {}
    final_scores = {}

    for idx, sim in zip(top_k, top_k_sim):
        cast_members = df.iloc[idx]["cast_names"][:10]
        for i, actor in enumerate(cast_members):
            weight = 3.0 if i < 5 else 1.0
            score = (float(sim) ** 2) * weight
            sim_scores[actor] = sim_scores.get(actor, 0.0) + score

    for actor, sim_score in sim_scores.items():
        dir_score = director_actor_counts.get((predicted_director, actor), 0.0)
        final_scores[actor] = sim_score + 0.1 * dir_score

    ranked = sorted(final_scores, key=final_scores.get, reverse=True)
    return ranked[:top_n]

def suggest_title(text, top_k):
    words = re.findall(r"[a-zA-Z']+", text.lower())
    words = [w for w in words if w not in STOPWORDS and len(w) > 3]
    freq = Counter(words)
    keyword = freq.most_common(1)[0][0].title() if freq else "Untitled"

    top_titles = [df.iloc[idx]["original_title"] for idx in top_k[:3]]

    for title in top_titles:
        if keyword.lower() in title.lower():
            return title

    if top_titles:
        return f"{keyword}: {top_titles[0]}"
    return keyword

def evaluate():
    with open("data/test_df.pkl", "rb") as f:
        test_df = pickle.load(f)

    correct = 0
    total_score = 0

    for i in range(len(test_df)):
        row = test_df.iloc[i]

        features, emb = build_classifier_features(row['overview'])
        sim_idx, sim = get_similar(emb)

        # predicted_director, _ = predict_director_emb(sim_idx, sim)
        # predicted_director, _ = predict_director_ml(features)
        predicted_director, _ = predict_director_hybrid(features, sim_idx, sim)
        predicted_cast = predict_cast(sim_idx, sim, predicted_director, top_n=15)

        actual_dir = row["director"]
        actual_cast = row["cast_names"]

        score = 0
        if predicted_director == actual_dir:
            score += 20
            correct += 1

        base_hits = 0
        bonus_hits = 0
        actual_cast_set = set(actual_cast)
        actual_top5 = set(actual_cast[:5])

        for actor in predicted_cast:
            if actor in actual_cast_set and base_hits < 5:
                base_hits += 1
            if actor in actual_top5 and bonus_hits < 5:
                bonus_hits += 1

        score += 10 * min(base_hits, 5)
        score += 5 * min(bonus_hits, 5)

        print(f"{row['original_title']}: {score}")
        total_score += score

    print(f"average score = {total_score / len(test_df):.2f}")
    print(f"director accuracy = {(correct / len(test_df)) * 100:.2f}%")

def predict_movie(text):
    features, emb = build_classifier_features(text)
    sim_idx, sim = get_similar(emb)

    predicted_dir, _ = predict_director_hybrid(features, sim_idx, sim)
    predicted_cast = predict_cast(sim_idx, sim, predicted_director=predicted_dir, top_n=15)
    predicted_title = suggest_title(text, sim_idx)

    print(text)
    print(f"Title suggestion: {predicted_title}")
    print(f"Director suggestion: {predicted_dir}")
    print(f"Cast suggestions: {', '.join(predicted_cast)}")

def main():
    if len(sys.argv) > 2:
        print("Usage: robotproducer.py [input.txt]")
        print("Evaluation: robotproducer.py")
    elif len(sys.argv) == 1:
        evaluate()
    else:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            text = f.read()
        predict_movie(text)


if __name__ == "__main__":
    main()