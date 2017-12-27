"""
output paper content to one txt file
"""
import os

os.chdir('E:\HDQ\Projects\Graduation')

papers_path = 'E:\HDQ\Projects\Graduation\output\\texts_without_main_text'
output_file_path = './output/papers_without_main_text_12_20.txt'

with open(output_file_path, 'w', encoding='utf-8') as outputf:
    count = 0
    for filename in os.listdir(papers_path):
        with open(os.path.join(papers_path, filename), encoding='utf-8') as inputf:
            for line in inputf.readlines():
                outputf.write(line.replace('\ufeff', ''))
            inputf.close()
        count += 1
        if count % 1000 == 0:
            print('Processed %d files!' % count)
    outputf.flush()
    outputf.close()
