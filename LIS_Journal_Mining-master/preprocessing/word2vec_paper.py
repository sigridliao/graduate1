from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
import os

os.chdir('E:\HDQ\Projects\Graduation')
raw_data = './output/train_with_papers_without_main_text_12_20.txt'
output_file = './output/paper_without_main_text_400_word2vec'
output_model_file = './output/paper_without_main_text_400.model'
vocabulary_file = './output/vocabulary'
# output_data_bin = './output/paper_without_main_text_400.bin'
print('Start Word2Vec...')
model = word2vec.Word2Vec(LineSentence(raw_data), size=400, window=10,
                          seed=1337, min_count=0, negative=0, hs=1,
                          workers=8, iter=15)
model.save(output_file)
print('generate Model File...')
model.wv.save_word2vec_format(output_model_file, fvocab=vocabulary_file, binary=False)
# print('generate Bin File...')
# model.wv.save_word2vec_format(output_data_bin, binary=True)
print('Finished!')
