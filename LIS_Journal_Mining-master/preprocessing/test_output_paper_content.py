"""
输出某标签下论文内容
"""
label_file_path = 'E:\HDQ\Projects\Graduation\\output\\doc_topic-15.txt'
papers_path = 'E:\HDQ\Resources\\texts_without_main_text_11_9'

ids = list()
count = 0
with open(label_file_path, encoding='utf-8') as f:
    for l in f.readlines():
        id_label = l.replace('\r', '').replace('\n', '').strip().split(':')
        label = int(id_label[1])
        if label == 15:
            count += 1
            ids.append(id_label[0])
    f.close()

for id in ids:
    paper_file_path = papers_path + '\\' + id + '.txt'
    with open(paper_file_path, encoding='utf-8') as f:
        print('The ID is: %s' % id)
        print(f.read().replace(' ', ''))
        print()
        f.close()

print('The Total Count is: %d' % count)

