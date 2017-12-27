"""
LDA Clustering for the clustering of LIS papers
"""

import lda
import lda.datasets
import os
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

path = 'E:\HDQ\Resources\\texts_without_main_text_11_7'

os.chdir('E:\HDQ\Projects\Graduation')

data = list()
titles = list()

ids = set()
id_file_path = 'E:\HDQ\Resources\papers\\1_12.txt'
with open(id_file_path, encoding='utf-8') as f:
    for l in f.readlines():
        l = l.replace('\n', '')
        ids.add(l)

for id in os.listdir(path):
    if id[:-4] in ids:
        with open(os.path.join(path, id), encoding='utf-8') as f:
            data.append(f.read().replace('\n', ' '))
        id = id[:-4]
        titles.append(id)


_count_vectorizer = CountVectorizer()
_X = _count_vectorizer.fit_transform(data)
vocab = list(_count_vectorizer.vocabulary_.keys())
count_vectorizer = CountVectorizer(vocabulary=vocab)
X = count_vectorizer.fit_transform(data)

for n_topics in range(3, 4):
    model = lda.LDA(n_topics=n_topics, n_iter=3000, random_state=1)
    model.fit(X)

    output_path = './output'

    topic_word = model.topic_word_
    n_top_words = 50
    file_path = output_path + '/topics-' + str(n_topics) + '.txt'
    with open(file_path, 'w', encoding='utf-8') as topic_file:
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
            print('Topic {}: {}'.format(i + 1, ' '.join(topic_words)))
            topic_file.write('Topic %d: ' % (i + 1))
            topic_file.write('\n')
            topic_file.write(' '.join(topic_words))
            topic_file.write('\n\n')
        topic_file.flush()
        topic_file.close()

    doc_topic = model.doc_topic_
    with open(output_path + '/doc_topic-' + str(n_topics) + '.txt', 'w', encoding='utf-8') as doc_topic_file:
        for i in range(len(titles)):
            doc_topic_file.write('%s:%s\n' % (titles[i], str(doc_topic[i].argmax() + 1)))
        doc_topic_file.flush()
        doc_topic_file.close()




