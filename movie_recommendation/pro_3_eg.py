#first we import the countervectorizer class that contains fit_transform methord 
from sklearn.feature_extraction.text import CountVectorizer
#here cosine_similarity is a methord itself so no need to call aobject
from sklearn.metrics.pairwise import cosine_similarity

text = ["London Paris London","Paris Paris London"]

#here we made a object cv of class CountVectorizer
cv = CountVectorizer()

#calling fit_transform on the object cv
count_matrix = cv.fit_transform(text)

#toarray is used to convert the data in a array
#print (count_matrix.toarray()) 

#calling cosine_similarity methord
similarity_scores = cosine_similarity(count_matrix)


print (similarity_scores)
#the result indicate that 1st - 1 means the 1st sentence is 100 similear to itself
# then the 0.8 indicate in the 1st box , that 1st sentence is 80% similar to the 2nd sentence

#in second box again the 1st 0.8 indicate that 2nd sentence is 80% similar to the 1st sentence
#and the 1 indicate that the second sentence is 100 similar to 2nd sentence 