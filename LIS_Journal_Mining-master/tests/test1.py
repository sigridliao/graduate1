import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


label_file_path = 'E:\HDQ\Projects\Graduation\\output\\doc_topic-15_1124.txt'
id_label_dict = dict()
with open(label_file_path, encoding='utf-8') as label_file:
    for line in label_file.readlines():
        line = line.strip('\n').strip('\ufeff')
        id_label = line.split(':')  # ['30', '2']
        id_label_dict[int(id_label[0])] = int(id_label[1])

text_files_path = 'E:\HDQ\Resources\\texts_without_main_text_11_9'
data = list()
labels = list()
for text_filenames in os.listdir(text_files_path):
    id = int(text_filenames[:-4])
    text_file_path = os.path.join(text_files_path, text_filenames)
    with open(text_file_path, encoding='utf-8') as text_file:
        data.append(text_file.read().replace('\n', ' '))
        labels.append(id_label_dict[id])

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=1)

count_vectorizer = CountVectorizer()
X_train_count = count_vectorizer.fit_transform(X_train, y_train)
X_test_count = count_vectorizer.transform(X_test)

k = 10000
feature_selector = SelectKBest(chi2, k=k)
X_train_chi2 = feature_selector.fit_transform(X_train_count, y_train)
X_test_chi2 = feature_selector.transform(X_test_count)

tfidf_transformer = TfidfTransformer(sublinear_tf=True)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_chi2, y_train)
X_test_tfidf = tfidf_transformer.transform(X_test_chi2)

classifier = MultinomialNB(alpha=1)
classifier.fit(X_train_tfidf, y_train)
y_predict = classifier.predict(X_test_tfidf)
print(classification_report(y_test, y_predict, digits=4))
