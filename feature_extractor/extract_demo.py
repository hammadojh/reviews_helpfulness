import pandas as pd
from extract_features import extract_all_features

data = pd.read_csv('../data/arabic_book_reviews.csv')
extract_all_features(data['English review'], min_doc_freq=2, saveto='arabic_books')