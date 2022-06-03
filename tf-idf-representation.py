from sklearn.feature_extraction.text import TfidfVectorizer
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

tfidf_vectorizer = TfidfVectorizer(vocabulary=vocab)
tfidf = tfidf_vectorizer.fit_transform(corpus).toarray()

sns.heatmap(tfidf,
            annot=True,
            cbar=False,
            xticklabels=vocab,
            yticklabels=[sentence for sentence in corpus]
            )

plt.show()
