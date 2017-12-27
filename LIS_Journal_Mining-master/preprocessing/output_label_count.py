"""
输出标签下论文数量
"""

label_file_path = 'E:\HDQ\Projects\Graduation\\output\\doc_topic-15_1124.txt'
papers_path = 'E:\HDQ\Resources\\texts_without_main_text_11_9'

label_count = dict()

with open(label_file_path, encoding='utf-8') as f:
    for l in f.readlines():
        id_label = l.replace('\r', '').replace('\n', '').strip().split(':')
        label = int(id_label[1])
        if label not in label_count.keys():
            label_count[label] = 1
        else:
            label_count[label] += 1
    f.close()

count = 0
for i in range(len(label_count.keys())):
    print('Count of cluster %d: %d' % (i + 1, label_count[i + 1]))
    count += label_count[i + 1]

print('\nThe Total Count is: %d' % count)