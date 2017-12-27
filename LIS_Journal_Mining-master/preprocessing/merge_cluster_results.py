"""
以file1为优先
合并聚类标签结果
"""
file1_path = 'E:\HDQ\Projects\Graduation\output\\iter2_part2.txt'
file2_path = 'E:\HDQ\Projects\Graduation\output\\doc_topic-15_iter1.txt'
merge_file_path = 'E:\HDQ\Projects\Graduation\output\\doc_topic-15_iter2.txt'
file1_ids = set()

with open(merge_file_path, 'w', encoding='utf-8') as mf:
    with open(file1_path, encoding='utf-8') as f1:
        for l in f1.readlines():
            mf.write(l)
            id_label = l.replace('\n', '').split(':')
            file1_ids.add(id_label[0])
        f1.close()
    with open(file2_path, encoding='utf-8') as f2:
        for l in f2.readlines():
            id_label = l.replace('\n', '').split(':')
            if id_label[0] not in file1_ids:
                mf.write(l)
        f2.close()
    mf.flush()
    mf.close()
