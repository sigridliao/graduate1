"""
输出特定标签的id
"""

label_file_path = 'E:\HDQ\Projects\Graduation\\output\\doc_topic-17.txt'

labels = {1, 12}

file_path = 'E:\HDQ\Resources\papers\\1_12.txt'

with open(file_path, 'w', encoding='utf-8') as newf:
    with open(label_file_path, encoding='utf-8') as f:
        for l in f.readlines():
            id_label = l.replace('\r', '').replace('\n', '').strip().split(':')
            label = int(id_label[1])
            id = id_label[0]
            if label in labels:
                newf.write(id)
                newf.write('\n')
        f.close()
    newf.flush()
    newf.close()
