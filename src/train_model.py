from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

df = pd.read_csv('data/product_reviews_full.csv')
df = df.dropna()

df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()

df['sentiment'] = df['sentiment'].astype('category')

df = df.drop(columns=['review_uuid', 'product_name', 'product_price'])

df['review_length'] = df['review_text'].astype(str).str.len()

x = df[['review_title','review_text', 'review_length']]
y = df['sentiment']

preprocessor = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(), "review_title"),
        ("text", TfidfVectorizer(), "review_text"),
        ("length", MinMaxScaler(), ["review_length"])
    ]
)

pipeline = Pipeline([
      ("preprocessing", preprocessor),
      ("classifier", RandomForestClassifier)
  ])

pipeline.fit(x, y)

joblib.dump(pipeline, "model/sentiment_model.pkl")

print("Model trained and saved!")