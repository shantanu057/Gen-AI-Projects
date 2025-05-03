from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import numpy as np

nltk.download('punkt')

word = ["viratkohli", "rohitsharma", "msdhoni", "ronaldo", "messi", "football"] #Add as much as you want

tokenized_sentence = [word_tokenize(data.lower()) for data in word]

# print("token", tokenized_sentence)

model = Word2Vec(sentences = tokenized_sentence, vector_size=100, window=5, min_count=1, workers=4)  # convert word to vector

#get embedding of specific word

word_embedding = model.wv["viratkohli"]  #It will only convert into embeddings consist of multiple binary codes seperated by space so it is difficult to find in the database.
print(word_embedding)
print("#####################################################################")
print("#####################################################################")
embedded_string = np.array2string(word_embedding, separator=',')   #It will convert into embeddings consist of multiple binary codes seperated by , so it can be stored in the matrix form and easy to find in the database.
print("embedded_string", embedded_string)