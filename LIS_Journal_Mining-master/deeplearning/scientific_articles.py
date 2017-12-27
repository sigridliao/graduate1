"""
The Scientific Article Data Class
Also Contain Relevant Test Models and Cases~
"""
from preprocessing.generate_label_dict import get_label_dict_lda
import os
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Embedding
from keras.layers import Convolution2D, MaxPooling2D, GlobalMaxPooling2D


class Article:
    """
    The Scientific Article Class
    """
    def __init__(self, article_data_path='', id_label_path=''):
        """
        Initialize the IDs, the Raw Data and the Labels

        Parameters
        ----------
        article_data_path: the path of the article data

        id_label_path: the path of the file which contains the id:label sequences

        Initialization Vars:
        --------------------
        data_collection: a dict collection with the whole scientific article data, contains
            ids (list(int)), raw data (list(list(word))) and labels (list(int))

        max_length_dict: a dict contains the max lengths of words of different sections
            0: title; 1: abstract; 2: keywords; 3: main text

        min_length_dict: a dict contains the min lengths of words of different sections
            0: title; 1: abstract; 2: keywords; 3: main text

        """
        print('\nInitializing the ARTICLES...\n')
        if article_data_path == '' or id_label_path == '':
            return
        self.data_collection = {
            'ids': list(),
            'raw_data': list(),
            'labels': list()
        }
        self.max_length_dict = {
            0: 0,
            1: 0,
            2: 0,
            3: 0
        }
        self.min_length_dict = {
            0: 999999,
            1: 999999,
            2: 999999,
            3: 999999
        }
        error_count = 0
        id_labels = get_label_dict_lda(id_label_path)
        for k in id_labels.keys():
            article_file_path = article_data_path + '\\' + str(k) + '.txt'
            with open(article_file_path, encoding='utf-8') as f:
                self.data_collection['ids'].append(k)
                text_seqs = list()
                lines = f.readlines()
                for i in range(len(lines)):
                    text_seq = lines[i].strip('\n')
                    text_seqs.append(text_seq)
                    lenwords = len(text_seq.split(' '))
                    if i == 0:
                        if lenwords > 25 or lenwords < 2:
                            print('ID: %d, Title: %s' % (k, lines[i]))
                            error_count += 1
                    if i == 1:
                        if lenwords > 250 or lenwords < 10:
                            print('ID: %d, Abstract: %s' % (k, lines[i]))
                            error_count += 1
                    if i == 2:
                        if lenwords > 10 or lenwords < 3:
                            print('ID: %d, Keywords: %s' % (k, lines[i]))
                            error_count += 1
                    if lenwords > self.max_length_dict[i]:
                        self.max_length_dict[i] = lenwords
                    elif lenwords < self.min_length_dict[i]:
                        self.min_length_dict[i] = lenwords
                self.data_collection['raw_data'].append(text_seqs)
                self.data_collection['labels'].append(id_labels[k])
        print('Initialization Finished! Collected %d ARTICLES! %d Errors Occured!' %
              (len(self.data_collection['raw_data']), error_count))

    def get_embed_data(self, embed_size, word2vec_path=''):
        """
        Get the word-embedded text data based on the raw text data

        Parameters
        ----------
        embed_size: word2vec embedding size

        word2vec_path: the file path of the word2vec file

        Return
        ------
        embed_data(nparray): shaped as [num_data, num_sections(max_section_lengths_of_words, embed_size)]

        """
        print('\nGetting Embedding Data From Path %s\n' % word2vec_path)
        embed_data = list()
        model = word2vec.Word2Vec.load(word2vec_path)
        # model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        print('Finished Loading Word2Vec Model!')
        # load each article text
        count = 0
        error_count = 0
        for raw_text in self.data_collection['raw_data']:
            embed_text = list()
            # load each section (title, abstract, keywords, main text)
            for i in range(len(raw_text)):
                embed_text_section = list()
                # load each word
                text_len = 0
                words = raw_text[i].split(' ')
                length_left = int((self.max_length_dict[i] - len(words)) / 2)
                for j in range(length_left):
                    embed_text_section.append(np.asarray([0.0] * embed_size))
                    text_len += 1
                for j in range(len(words)):
                    try:
                        word = words[j].replace('\n', '').replace('\ufeff', '')
                        embed_text_section.append(np.asarray(model.wv[word]))
                        text_len += 1
                    except:
                        print('Word %s Not In Word2Vec Vocabulary!' % words[j])
                        error_count += 1
                        continue
                while text_len <= self.max_length_dict[i]:
                    embed_text_section.append(np.asarray([0.0] * embed_size))
                    text_len += 1
                embed_text.append(np.asarray(embed_text_section))
            embed_data.append(np.asarray(embed_text))
            count += 1
            if count % 1000 == 0:
                print('processed %d articles' % count)
        print('\nFinished Getting Embedding Data! %d Errors Occured!\n' % error_count)
        return np.asarray(embed_data)


if __name__ == '__main__':
    pass


