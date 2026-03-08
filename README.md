# Content-Based Movie Recommender System

A machine learning project that recommends movies based on content similarity using **Natural Language Processing** and **Cosine Similarity**.

---

## Overview

This recommender system analyzes movie metadata — genres, keywords, cast, crew, and overview — to find and suggest movies similar to a given title. It is built on the **TMDB 5000 Movies Dataset** and uses a **Bag of Words** model combined with cosine similarity to measure how alike two movies are.

---

## How It Works

```
DATA  ──►  PREPROCESSING  ──►  MODEL  ──►  RECOMMENDATIONS
```

1. **Data Collection** — Merges two datasets: `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`
2. **Feature Selection** — Extracts key columns: `genres`, `keywords`, `overview`, `cast` (top 3), `crew` (director only)
3. **Preprocessing**
   - Parses JSON-like strings using `ast.literal_eval`
   - Removes spaces within multi-word tags to avoid ambiguity (e.g. `Sam Worthington` → `SamWorthington`)
   - Combines all features into a single `tags` column
   - Applies **Porter Stemming** to reduce words to their root form
   - Converts to lowercase
4. **Vectorization** — Uses `CountVectorizer` (Bag of Words) with top 5000 features, removing English stop words
5. **Similarity** — Computes **Cosine Similarity** between all movie vectors
6. **Recommendation** — Returns the top 5 most similar movies for any given title

---

## Project Structure

```
movie-recommender-system/
│
├── Movie_Recommender.ipynb     # Main notebook
├── requirements.txt            # Dependencies
└── README.md
```

> **Note:** The datasets `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` are required to run this notebook.  
> Download them from [Kaggle – TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `ast` | Parsing JSON-like strings |
| `nltk` | Porter Stemming |
| `scikit-learn` | CountVectorizer, Cosine Similarity |

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/your-username/movie-recommender-system.git
cd movie-recommender-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the datasets from Kaggle and place them in the project folder

# 4. Launch Jupyter Notebook
jupyter notebook Movie_Recommender.ipynb
```

---

## Example Output

```python
recommend('Avatar')
# → Titan A.E.
# → Independence Day
# → Ender's Game
# → John Carter
# → Aliens vs Predator: Requiem
```

---

## Types of Recommender Systems

| Type | Description |
|---|---|
| **Content-Based** | Recommends based on item features (this project) |
| Collaborative Filtering | Recommends based on user behavior |
| Hybrid | Combination of both |

---

## 👥 Authors

- **Anjor Patil** 
- **Aditi Lokhande** 
