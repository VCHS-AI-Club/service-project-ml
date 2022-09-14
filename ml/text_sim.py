from typing import List
# Import pandas for data manipulation
import pandas as pd
# Import TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
# Import cosine similarity
from sklearn.metrics.pairwise import cosine_similarity


new_entry: str = "California Coastal"

# simulated opportunity list of database
opp: List[str] = [
    "California Coastal Cleanup Day Various County Park locations Join millions of Americans on Coastal Cleanup Day to remove litter in and around our local waterways.",
    "Martial Cottle Fall Festival Martial Cottle Park Assist with a variety of activity stations which include cattle corral, kids crafts, truck/tractor area, and parking lot monitors",
    "Day on the Bay Alviso Marina County Park Volunteers needed to assist with event setup; event activity support; parking management; on-site activity registration; and/or event tear down."
]

opp.append(new_entry)

tfidf = TfidfVectorizer(analyzer='word',
                      token_pattern=r'\w{1,}',
                      ngram_range=(1, 3), 
                      stop_words = 'english')

tfidf_matrix = tfidf.fit_transform(opp)

print(tfidf_matrix.shape)

similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# taking the first one for example  
similarity_scores = list(enumerate(similarity_matrix[len(opp)-1]))
sorted_similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

print(sorted_similarity_scores)