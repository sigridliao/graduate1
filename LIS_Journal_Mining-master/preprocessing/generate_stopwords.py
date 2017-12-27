"""
生成停用词
"""
import os

os.chdir('E:\HDQ\Projects\Guaduation')

stopwords_path = './res/stopwords'
stopwords_file = 'stopwords_17_10_30.txt'
file_list = os.listdir(stopwords_path)
print('Stopwords Path: %s' % stopwords_path)
print('Stopwords File: %s' % stopwords_file)
print('Start Collecting Stopwords...')
stopwords = set()
with open(stopwords_path + '/' + stopwords_file, 'w', encoding='utf-8') as swf:
    for filename in file_list:
        if filename == stopwords_file:
            continue
        file_path = os.path.join(stopwords_path, filename)
        print('Loading File %s' % filename)
        with open(file_path, encoding='utf-8') as f:
            for w in f.readlines():
                w = w.strip().replace('\n', '')
                stopwords.add(w)
            f.close()
    print('Collected %d Stopwords!' % len(stopwords))
    for stopword in stopwords:
        swf.write(stopword)
        swf.write('\n')
    swf.flush()
    swf.close()
print('Finished!')

