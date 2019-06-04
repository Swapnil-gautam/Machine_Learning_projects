import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
## helper function use them when needed ###
def get_title_from_index(index):
 	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
 	return df[df.title == title]["index"].values[0]

#########################################

##Step 1 : Read CSV Files
df = pd.read_csv("movie_dataset.csv")
#print (df.head())

##Step 2: Selet Features
features = ['keywords','cast','genres','director']

##Step 3: Creat a column in DF which combines all selected features

###the fillna comand will fill all the blank places where it was Nan with a blank space
for feature in features:
	df[feature] = df[feature].fillna('')



def combine_features(row):
	try:
		return row['keywords'] +" "+row['cast']+" "+row['genres']+" "+row["director"]
	except:
		print("Error:",row)

###apply is used to apply combine_features() function
df["combined_features"] = df.apply(combine_features,axis=1)	

#print(df["combined_features"].head())

##step 4:create count matrix from this new combined column  
cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])

##step5: compute the cosine similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)
movie_user_likes ="The Avengers"

## step 6: get index of this movie from its title
movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(cosine_sim[movie_index]))

## step 7 : get a list of similar movies in descending order of similarity score
sorted_similar_movies = sorted(similar_movies,key= lambda x:x[1],reverse=True) 

## step 8: Print titles of first 50 movies
i =0
for movie in sorted_similar_movies:
	print(get_title_from_index(movie[0]))
	i=i+ 1
	if i>50:
		break