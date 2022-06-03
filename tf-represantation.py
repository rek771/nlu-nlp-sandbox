# Генерация унитарного или бинарного представления с помощью scikit-learn
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

corpus = ['Whisky cheaper in Alabama.',
          'Russian whisky is cheaper because it`s vodka.',
          'Russian whisky is cheaper than Alabama']
vocab = []

for tmp in corpus:
    for word in tmp.split(' '):
        vocab.append(word.replace('.', '').lower())
vocab = list(set(vocab))

one_hot_vectorizer = CountVectorizer(binary=True, vocabulary=vocab)
one_hot = one_hot_vectorizer.fit_transform(corpus).toarray()

sns.heatmap(one_hot,
            annot=True,
            cbar=False,
            xticklabels=vocab,
            yticklabels=[sentence for sentence in corpus]
            )

plt.show()
