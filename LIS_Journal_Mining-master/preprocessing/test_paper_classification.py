"""
聚类结果自分类测试
"""

from preprocessing.generate_label_dict import get_label_dict_lda
import lda
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np
import os
import keras

label_file_path = 'E:\HDQ\Projects\Graduation\\output\\'
papers_path = 'E:\HDQ\Resources\\texts_with_main_text_11_29'

# id_labels = dict()
# with open(label_file_path, encoding='utf-8') as f:
#     count = 0
#     for l in f.readlines():
#         if count == 30:
#             break
#         id_label = l.replace('\r', '').replace('\n', '').strip().split(':')
#         id_labels[int(id_label[0])] = int(id_label[1])
#     f.close()

n_topics = 15
print('\nn-topics is %d...' % n_topics)
# file_path = label_file_path + str(n_topics) + '_doc_topic.txt'
file_path = label_file_path + 'doc_topic-' + str(n_topics) + '_1124.txt'
id_labels = get_label_dict_lda(file_path)

data = list()
label = list()
for id in id_labels.keys():
    paper_file_path = papers_path + '\\' + str(id) + '.txt'
    with open(paper_file_path, encoding='utf-8') as f:
        data.append(str(id) + ' ' + f.read().replace('\n', ' '))
        label.append(id_labels[id])


kf = KFold(n_splits=10)
count = 0
avg_accuracy = 0.0
avg_precision = 0.0
avg_recall = 0.0
# for train, tests in kf.split(data):
for k_times in range(10):
    print('\nThe Count is: %s' % str(count))
    # X_train, X_test, y_train, y_test = np.array(data)[train], np.array(data)[tests], np.array(label)[train], np.array(label)[tests]
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.4, random_state=1)

    train_ids = list()
    test_ids = list()
    X_train_new = list()
    X_test_new = list()
    for i in range(len(X_train)):
        first_space_index = X_train[i].index(' ')
        train_ids.append(int(X_train[i][:first_space_index]))
        X_train_new.append(X_train[i][first_space_index + 1:])
    for i in range(len(X_test)):
        first_space_index = X_test[i].index(' ')
        test_ids.append(int(X_test[i][:first_space_index]))
        X_test_new.append(X_test[i][first_space_index + 1:])


    count_vectorizer = CountVectorizer()
    X_train_count = count_vectorizer.fit_transform(X_train_new, y_train)
    X_test_count = count_vectorizer.transform(X_test_new)

    # lda_model = lda.LDA(n_topics=n_topics, n_iter=3000, random_state=1)
    # X_train_lda = lda_model.fit_transform(X_train_count, y_train)
    # X_test_lda = lda_model.transform(X_test_count)

    k = 2500 * (k_times + 1)
    # k = 20000
    print('\nThe k is: %d' % k)
    feature_selector = SelectKBest(chi2, k=k)
    X_train_chi2 = feature_selector.fit_transform(X_train_count, y_train)
    X_test_chi2 = feature_selector.transform(X_test_count)
    # for i in feature_selector.get_support(indices=True):
    #     print(count_vectorizer.get_feature_names()[i])


    tfidf_transformer = TfidfTransformer(sublinear_tf=True)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_chi2, y_train)
    X_test_tfidf = tfidf_transformer.transform(X_test_chi2)

    # classifier = MultinomialNB(alpha=1, fit_prior=False)
    # classifier = LinearSVC(random_state=1, max_iter=1000, C=0.5, class_weight='balanced')
    # classifier = KNeighborsClassifier(n_neighbors=70, weights='distance')
    classifier = MLPClassifier(hidden_layer_sizes=(400, ), max_iter=1000)
    classifier.fit(X_train_tfidf, y_train)
    y_predict = classifier.predict(X_test_tfidf)

    print(classification_report(y_test, y_predict, digits=4))
    # precision = precision_score(y_test, y_predict, average=None)
    # accuracy = accuracy_score(y_test, y_predict)
    # recall = recall_score(y_test, y_predict, average=None)
    #
    # avg_accuracy += accuracy
    # avg_precision += precision
    # avg_recall += recall
    # save = pd.DataFrame({
    #     'test_ids': test_ids,
    #     'label_ids': y_test,
    #     'predicted_label_ids': y_predict
    # })
    # os.chdir('E:\HDQ\Projects\Graduation')
    # save.to_csv('./output/classification_' + str(count) + '.csv', index=False, encoding='utf-8')
    # count += 1

# avg_accuracy /= count
# avg_precision /= count
# sum = 0.0
# for p in avg_precision:
#     sum += p
# avg_single_precision = sum / 15
# avg_recall /= count
# sum = 0.0
# for r in avg_recall:
#     sum += r
# avg_single_recall = sum / 15
#
# print()
# print('The average precision is: %s' % str(avg_single_precision))
# print('The average recall is: %s' % str(avg_single_recall))
# print('The average accuracy is: %s' % str(avg_accuracy))
