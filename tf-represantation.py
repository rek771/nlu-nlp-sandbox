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
vocab = set(vocab)

one_hot_vectorizer = CountVectorizer(binary=True)
one_hot = one_hot_vectorizer.fit_transform(corpus).toarray()

sns.heatmap(one_hot,
            annot=True,
            cbar=False,
            xticklabels=vocab,
            yticklabels=[f"Предложение {num}" for num in range(1, len(corpus) + 1)]
            )

plt.show()
