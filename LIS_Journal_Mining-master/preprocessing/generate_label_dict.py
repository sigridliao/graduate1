"""
通过聚类结果
生成标签dict
"""

import os

def get_label_dict_mallet(label_file_path='E:\CYS\\topics\\doc-topics-20.txt'):
    print('\nGenerating Label Dict for MALLET...')
    label_dict = {}
    with open(label_file_path, encoding='utf-8') as f:
        count = 0
        for l in f.readlines():
            l = l.replace('\r', '').replace('\n', '').strip('\ufeff')
            id_labels = l.split('\t')
            if count != 0 and len(id_labels) == 4:
                labels = [-1, -1, -1]
                labels[0] = int(id_labels[1])
                labels[1] = int(id_labels[2])
                labels[2] = int(id_labels[3])
                label_dict[int(id_labels[0])] = labels
            count += 1
            if count % 1000 == 0:
                print(id_labels[0], id_labels[1], id_labels[2], id_labels[3])
                print('Processed %d papers...' % count)
        f.close()
    print('Finished!')
    return label_dict

def get_label_dict_lda(label_file_path=''):
    # print('\nGenerating Label Dict for PYTHON LDA...')
    label_dict = {}
    with open(label_file_path, encoding='utf-8') as f:
        count = 0
        for l in f.readlines():
            try:
                count += 1
                id_label = l.replace('\r', '').replace('\n', '').strip('\ufeff').split(':')
                label = int(id_label[1])
                label_dict[int(id_label[0])] = label
            except:
                print('Error at line %d' % count)
                exit(0)
            # if count % 1000 == 0:
                # print(id_label[0], id_label[1])
                # print('Processed %d papers...' % count)
        f.close()
    # print('Finished!')
    return label_dict