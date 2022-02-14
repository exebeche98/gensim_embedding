import pandas as pd
import numpy as np

df = pd.read_csv('IMDB Dataset.csv')
sentiment_type = np.unique(df['sentiment'])
level_map = {'positive': 1, 'negative': 0}
df['sentiment'] = df['sentiment'].map(level_map)
df['sentiment'] = df['sentiment'].astype(int)

import nltk
import re

def preprocessor(text):# удаляет знаки препинания, цифры, приводит всё в нижний регистр
    text = re.sub('<[^>]*>', '', text)
    text = re.sub("\d", '', text)# удалить цифры
    text = re.sub('[\W]+', ' ', text.lower())# +\
    return text

def delStopWords(text):# удаление стоп слов
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = nltk.tokenize.word_tokenize(text)
    words = [w for w in tokens if w not in stopwords]
    
    return ' '.join(list(words))

def denoise_text(text):
    text = preprocessor(text)
    text = delStopWords(text)
    return text

# Применяем обработку к набору данных
df['review']=df['review'].apply(denoise_text)

###############################################################################

from sklearn.model_selection import train_test_split   
X_train, X_test, y_train, y_test = train_test_split(
    df.review.values,
    df.sentiment.values,
    test_size=0.05,
    stratify=df.sentiment.values,
    shuffle=True,
    random_state=42)  

# Не следует обучать word2vec на всей выборке, используем для этого только обучающую
sent = []
for i in X_train:
    sent.append(i.split())

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 200  # We will cut reviews after 200 words
max_words = 10000  # We will only consider the top 10,000 words in the dataset

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

###############################################################################
# Обучим собственный word2veс и используем его в качестве слоя  embedding
from gensim.models import Word2Vec

embedding_dim = 256

w2v_model = Word2Vec(sentences = sent,
                     min_count = 5,
                     window = 5,
                     vector_size = embedding_dim)

###############################################################################
# Прежде чем использовать word2veс необходимо в качестве слоя embedding следует
# убедиться, что он разумно отражает корпус текстов в векторное пространство
w2v_model.wv.most_similar(positive=['good']) # вывести слова наиболее схожие с 'good'
w2v_model.wv.most_similar(positive=['bad']) # вывести слова наиболее схожие с 'bad'
w2v_model.wv.similarity("david", 'movie') # вывести cхожесть слов 'david' и 'movie'
w2v_model.wv.similarity("movie", 'film') # вывести cхожесть слов 'film' и 'movie'
w2v_model.wv.doesnt_match(['good', 'bad', 'love']) # вывести лишние слово из списка
# алгебраические операции со словами ребёнок + возраст - игрушка 
w2v_model.wv.most_similar(positive=["child", "age"], negative=['toy'], topn=3)
###############################################################################
vectors = w2v_model.wv
# Функция получения матри
def build_emedding_matrix(vectors, embedding_dim, word_index):
    vocab = vectors.key_to_index
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            if word in vocab:
                embedding_vector = vectors[word]
                if embedding_vector is not None:
                    # Слова не найденные в векторном представлении заменяются нулевыми векторами
                    embedding_matrix[i] = embedding_vector
            else:
                print(f"{word} is not in vocab")
    return embedding_matrix

embedding_matrix = build_emedding_matrix(vectors, embedding_dim, word_index)
###############################################################################
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

def build_embedding_model(embedding_matrix, maxlen):
    model = Sequential()
    model.add(Embedding(embedding_matrix.shape[0],
                        embedding_matrix.shape[1],
                        input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model
model = build_embedding_model(embedding_matrix, maxlen)
    
###############################################################################
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(X_test, y_test))
del w2v_model
###############################################################################
# Gensim позволяет загружать предобученные векторные представления слов

import gensim.downloader
print(list(gensim.downloader.info()['models'].keys()))# Список возможных представлений
# Загрузим предварительно обученное векторное представление и обучим нейронную сеть
# В данном примере загружается word2vec с размером вектора 300 обученный на новостях google
wv = gensim.downloader.load('word2vec-google-news-300')
embedding_dim = 300
embedding_matrix = build_emedding_matrix(wv, embedding_dim, word_index)

model = build_embedding_model(embedding_matrix, maxlen)
    

history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(X_test, y_test))

del wv
###############################################################################
# Загрузка fasttext
fasttext_vectors = gensim.downloader.load('fasttext-wiki-news-subwords-300')
embedding_dim = 300
embedding_matrix = build_emedding_matrix(fasttext_vectors, embedding_dim, word_index)

model = build_embedding_model(embedding_matrix, maxlen)
    

history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(X_test, y_test))

del fasttext_vectors
###############################################################################
# Загрузка glove
embedding_dim = 25
glove_vectors = gensim.downloader.load('glove-twitter-25')
embedding_matrix = build_emedding_matrix(glove_vectors, embedding_dim, word_index)

model = build_embedding_model(embedding_matrix, maxlen)
    

history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(X_test, y_test))







