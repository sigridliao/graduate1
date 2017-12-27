from deeplearning.scientific_articles import Article
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Embedding, LSTM
from keras.layers import Convolution2D, MaxPooling2D, GlobalMaxPooling2D
import os
import keras
from keras.models import Model
from keras.utils import np_utils
from keras.models import load_model
from keras.layers.noise import GaussianNoise, GaussianDropout
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization


def get_reshaped_embed_data(embed_data):
    """
    Reshape the data for deeplearning model input (CNN)

    Parameters
    ----------
    embed_data: the embedding data

    Return
    ------
    reshaped_embed_data: shaped as [num_sections, num_data(max_section_lengths_of_words, 1, embed_size)]
    """

    print('\nReshaping Data...\n')
    title_data = list()
    abstract_data = list()
    keywords_data = list()
    for i in range(embed_data.shape[0]):
        for j in range(embed_data.shape[1]):
            if i == 0:
                print(embed_data[i][j].shape)
            reshaped_data = embed_data[i][j].reshape(embed_data[i][j].shape[0], 1, embed_data[i][j].shape[1])
            if j == 0:
                title_data.append(reshaped_data)
            elif j == 1:
                abstract_data.append(reshaped_data)
            elif j == 2:
                keywords_data.append(reshaped_data)
        if (i + 1) % 1000 == 0:
            print('Reshaped %d Data!' % (i + 1))
    title_data = np.asarray([title_data[i] for i in range(len(title_data))])
    abstract_data = np.asarray([abstract_data[i] for i in range(len(abstract_data))])
    keywords_data = np.asarray([keywords_data[i] for i in range(len(keywords_data))])
    print('Finished Reshaping Data!')
    reshaped_embed_data = [title_data, abstract_data, keywords_data]
    return reshaped_embed_data


def get_deeplearning_model(reshaped_embed_data, num_classes):
    """
    Get the deep-learning model for scientific articles

    Parameters
    ----------
    reshaped_embed_data: the reshaped embedding data of raw texts (train or text)

    num_classes: the number of classes

    """
    print('\nBuilding Deeplearning Model...\n')

    # Load Shape & Input
    title_shape = reshaped_embed_data[0][0].shape
    abstract_shape = reshaped_embed_data[1][0].shape
    keywords_shape = reshaped_embed_data[2][0].shape

    # title_input = reshaped_embed_data[0]
    # abstract_input = reshaped_embed_data[1]
    # keywords_input = reshaped_embed_data[2]

    title_input = Input(title_shape, dtype='float32', name='title_input')
    abstract_input = Input(abstract_shape, dtype='float32', name='abstract_input')
    keywords_input = Input(keywords_shape, dtype='float32', name='keywords_input')

    # Title Part
    title_input_bn = BatchNormalization()(title_input)
    title_conv_1_1 = Convolution2D(input_shape=title_shape,
                                   filters=10,
                                   kernel_size=(2, 1),
                                   strides=(1, 1),
                                   activation='relu',
                                   data_format='channels_last')(title_input_bn)
    title_pool_1_1 = GlobalMaxPooling2D()(title_conv_1_1)
    title_conv_1_2 = Convolution2D(input_shape=title_shape,
                                   filters=10,
                                   kernel_size=(3, 1),
                                   strides=(1, 1),
                                   activation='relu',
                                   data_format='channels_last')(title_input)
    title_pool_1_2 = GlobalMaxPooling2D()(title_conv_1_2)
    title_merge = keras.layers.concatenate([title_pool_1_1, title_pool_1_2])
    title_merge_bn = BatchNormalization()(title_merge)
    title_output = Dense(64, activation='sigmoid', kernel_regularizer=l2(0.01))(title_merge_bn)

    # Abstract Part
    abstract_input_bn = BatchNormalization()(abstract_input)
    abstract_conv_1_1 = Convolution2D(input_shape=abstract_shape,
                                      filters=20,
                                      kernel_size=(20, 1),
                                      strides=(1, 1),
                                      activation='relu',
                                      data_format='channels_last')(abstract_input_bn)
    abstract_pool_1_1 = MaxPooling2D(pool_size=(2, 1))(abstract_conv_1_1)
    abstract_conv_2_1 = Convolution2D(filters=10,
                                      kernel_size=(10, 1),
                                      strides=(1, 1),
                                      activation='relu',
                                      data_format='channels_last')(abstract_pool_1_1)
    abstract_pool_2_1 = MaxPooling2D(pool_size=(3, 1))(abstract_conv_2_1)
    abstract_conv_3_1 = Convolution2D(filters=10,
                                      kernel_size=(3, 1),
                                      strides=(2, 1),
                                      activation='relu',
                                      data_format='channels_last')(abstract_pool_2_1)
    abstract_pool_3_1 = GlobalMaxPooling2D()(abstract_conv_3_1)
    abstract_conv_3_2 = Convolution2D(filters=10,
                                      kernel_size=(4, 1),
                                      strides=(2, 1),
                                      activation='relu',
                                      data_format='channels_last')(abstract_pool_2_1)
    abstract_pool_3_2 = GlobalMaxPooling2D()(abstract_conv_3_2)
    abstract_conv_3_3 = Convolution2D(filters=10,
                                      kernel_size=(5, 1),
                                      strides=(2, 1),
                                      activation='relu',
                                      data_format='channels_last')(abstract_pool_2_1)
    abstract_pool_3_3 = GlobalMaxPooling2D()(abstract_conv_3_3)
    abstract_conv_3_4 = Convolution2D(filters=10,
                                      kernel_size=(6, 1),
                                      strides=(2, 1),
                                      activation='relu',
                                      data_format='channels_last')(abstract_pool_2_1)
    abstract_pool_3_4 = GlobalMaxPooling2D()(abstract_conv_3_4)
    abstract_merge = keras.layers.concatenate([abstract_pool_3_1, abstract_pool_3_2,
                                               abstract_pool_3_3, abstract_pool_3_4])
    abstract_merge_bn = BatchNormalization()(abstract_merge)
    abstract_output = Dense(512, activation='sigmoid', kernel_regularizer=l2(0.01))(abstract_merge_bn)

    # Keywords Part
    keywords_input_bn = BatchNormalization()(keywords_input)
    keywords_flatten = Flatten()(keywords_input_bn)
    keywords_output = Dense(64, activation='sigmoid', kernel_regularizer=l2(0.01))(keywords_flatten)

    # Merge and Output
    merge_layer = keras.layers.concatenate([title_output, abstract_output, keywords_output])
    merge_layer_bn = BatchNormalization()(merge_layer)
    dense_1 = Dense(256, activation='sigmoid', kernel_regularizer=l2(0.01))(merge_layer_bn)
    dropout_1 = GaussianDropout(0.25)(dense_1)
    dense_2 = Dense(32, activation='sigmoid', kernel_regularizer=l2(0.01))(dropout_1)
    dropout_2 = GaussianDropout(0.25)(dense_2)
    total_output = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01), name='total_output')(dropout_2)

    # Print Model Summary
    dl_model = Model(inputs=[title_input, abstract_input, keywords_input], outputs=[total_output])
    print(dl_model.summary())
    print('\nDeeplearning Model Building Completed...\n')
    return dl_model

if __name__ == '__main__':
    os.chdir('E:/HDQ/Projects/Graduation')

    _article_data_path = './output/texts_without_main_text'
    _id_label_path = './output/doc_topic-15_1124.txt'
    scientific_articles = Article(article_data_path=_article_data_path,
                                  id_label_path=_id_label_path)
    _word2vec_path = './output/paper_without_main_text_400_word2vec'
    scientific_articles_embed_data = scientific_articles.get_embed_data(400, word2vec_path=_word2vec_path)
    X_train, X_test, y_train, y_test = train_test_split(scientific_articles_embed_data,
                                                        scientific_articles.data_collection['labels'],
                                                        test_size=0.4,
                                                        random_state=1)
    X_Reshaped = get_reshaped_embed_data(scientific_articles_embed_data)
    X_train_Reshaped = get_reshaped_embed_data(X_train)
    X_test_Reshaped = get_reshaped_embed_data(X_test)
    y = scientific_articles.data_collection['labels']
    for i in range(len(y)):
        y[i] -= 1
    for i in range(len(y_train)):
        y_train[i] -= 1
    for i in range(len(y_test)):
        y_test[i] -= 1
    Y = np_utils.to_categorical(y, 15)
    Y_train = np_utils.to_categorical(y_train, 15)
    Y_test = np_utils.to_categorical(y_test, 15)

    model = get_deeplearning_model(X_Reshaped, 15)

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    model.fit({'title_input': X_train_Reshaped[0],
               'abstract_input': X_train_Reshaped[1],
               'keywords_input': X_train_Reshaped[2]},
              {'total_output': Y_train},
              batch_size=32,
              nb_epoch=30,
              validation_data=({'title_input': X_test_Reshaped[0],
                                'abstract_input': X_test_Reshaped[1],
                                'keywords_input': X_test_Reshaped[2]},
                               {'total_output': Y_test},))

    model.save('./output/nn_model/dl_model_171215.h5')
    score = model.evaluate({'title_input': X_test_Reshaped[0],
                            'abstract_input': X_test_Reshaped[1],
                            'keywords_input': X_test_Reshaped[2]},
                           {'total_output': Y_test},)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
