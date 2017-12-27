"""
每次合并txt里面的id跟label都有\ufeff字符搞事
贼鸡儿烦
检查重复id，后续手动去掉
"""
path = 'E:\HDQ\Projects\Graduation\output\doc_topic-15_iter2.txt'

ids = set()

print('\nChecking Duplicate ID...')

with open(path, encoding='utf-8') as f:
    for l in f.readlines():
        id = int(l.split(':')[0].strip('\ufeff'))
        if id in ids:
            print('Duplicate ID %d!' % id)
        else:
            ids.add(id)

print('Finished!')
