"""
生成停用前缀
"""

import os

os.chdir('E:\HDQ\Projects\Guaduation')
with open('./res/stopwords/stop_prefixes.txt', 'w', encoding='utf-8') as sp:
    prefixes_len1 = set()
    with open('./res/stopwords/punctuation_new.txt', encoding='utf-8') as pn:
        for l in pn.readlines():
            prefixes_len1.add(l[0])
        pn.close()
    for c in [chr(i) for i in range(32, 127)]:
        prefixes_len1.add(c)
    print('The length of stop prefixes set is %d' % len(prefixes_len1))
    for prefix in prefixes_len1:
        sp.write(prefix)
        sp.write('\n')
    sp.flush()
    sp.close()
