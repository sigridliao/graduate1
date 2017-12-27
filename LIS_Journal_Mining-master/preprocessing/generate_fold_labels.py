"""
生成（更新）某一fold的预测结果（按照分类器结果）
"""

import os
import pandas as pd

os.chdir('E:\HDQ\Projects\Graduation')

for fold_id in range(10):
    csv_file_path = './output/classification_' + str(fold_id) + '.csv'
    print('The CSV PATH is: %s' % csv_file_path)
    output_txt_path = './output/classification_' + str(fold_id) + '.txt'
    print('The TXT PATH is: %s' % output_txt_path)

    df = pd.read_csv(csv_file_path, encoding='utf-8')

    diff_df = df.loc[df.predicted_label_ids != df.label_ids]

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for _, row in diff_df.iterrows():
            f.write(str(row.test_ids))
            f.write(':')
            f.write(str(row.predicted_label_ids))
            f.write('\n')
        f.flush()
        f.close()

print('Finished!')
