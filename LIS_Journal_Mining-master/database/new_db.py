from sqlalchemy import Column, String, TEXT, INTEGER, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from preprocessing.generate_label_dict import get_label_dict_lda
import os
import jieba

# 创建对象的基类:
BaseModel = declarative_base()  # 建了一个 BaseModel 类，这个类的子类可以自动与一个表关联。


# 定义paper对象:
class Paper(BaseModel):
    # 表的名字:
    tables = ['paper_process_new', 'paper_2017_11_5', 'paper_2017_11_9']
    __tablename__ = tables[2]


    # 表的结构:
    id = Column(INTEGER(), primary_key=True)
    filename = Column(String(255))
    journal = Column(String(255))
    year = Column(String(4))
    issue = Column(String(2))
    title = Column(String(255))
    abstract = Column(TEXT(255))
    keyword = Column(String(255))
    label_id = Column(INTEGER())
    title_seg = Column(String(255))
    abstract_seg = Column(TEXT(255))
    keyword_seg = Column(String(255))
    main_text_seg = Column(TEXT(255))

    # label_1 = Column(INTEGER())
    # label_2 = Column(INTEGER())
    # label_3 = Column(INTEGER())


    # main_text = Column(TEXT(255))
    # section = Column(TEXT(255))
    # content_clean = Column(TEXT(255))
    # section_seg = Column(String(1000))
    # para_first = Column(TEXT(255))
    # para_mid = Column(TEXT(255))
    # para_last = Column(TEXT(255))

engine = create_engine('mysql+mysqlconnector://root:123456@localhost:3306/paper?charset=utf8mb4', echo=False)
DBSession = sessionmaker(bind=engine)


# 清洗字段abstract
def clean_abstract():
    session = DBSession()
    papers = session.query(Paper).all()
    print('\nCleaning the field ABSTRACT...')
    for paper in papers:
        paper.abstract = paper.abstract.replace(' ', '').replace('　', '')
    session.commit()
    print('Finished!')

# 清洗字段keywords
def clean_keywords():
    session = DBSession()
    papers = session.query(Paper).all()
    print('\nCleaning the field KEYWORDS...')
    for paper in papers:
        keywords = list()
        for kw in paper.keyword_seg.split(' '):
            if kw not in {''}:
                keywords.append(kw)
        paper.keyword_seg = ' '.join(keywords)
    session.commit()
    print('Finished!')

# 清洗正文的分词结果
def clean_main_text_seg(journal, stop_prefixes=set()):
    print('\nCleaning MAIN_TEXT_SEG of journal %s...' % journal)
    print('The size of the stop prefixes is %d' % len(stop_prefixes))
    session = DBSession()
    papers = session.query(Paper).filter(Paper.journal == journal).all()
    count = 0
    for paper in papers:
        if paper.main_text_seg == '' or paper.main_text_seg is None:
            continue
        words = paper.main_text_seg.split(' ')
        cleaned_main_text_seg = ''
        i = 0
        while True:
            if i >= len(words):
                break
            if len(words[i]) == 0:
                i += 1
                continue
            if words[i][0] in stop_prefixes or len(words[i]) < 2 or words[i].startswith('中图分类号') \
                    or words[i].startswith('标识码'):
                i += 1
                continue
            break
        for j in range(i, len(words)):
            if len(words[j]) != 0:
                cleaned_main_text_seg += words[j]
                cleaned_main_text_seg += ' '
        cleaned_main_text_seg = cleaned_main_text_seg.strip()
        paper.main_text_seg = cleaned_main_text_seg
        count += 1
        if count % 500 == 0:
            print('Processed %d Papers!' % count)
    session.commit()
    print('Finished!')


# 深度清洗正文的分词结果（第二步清洗，去除一些多余英文、中文停用词以及数字）
def clean_main_text_seg_deep(journal, stop_prefixes=set(), stopwords=set()):
    print('\nCleaning MAIN_TEXT_SEG of journal %s...' % journal)
    session = DBSession()
    papers = session.query(Paper).filter(Paper.journal == journal).all()
    count = 0
    for paper in papers:
        if paper.main_text_seg == '' or paper.main_text_seg is None:
            continue
        words = paper.main_text_seg.split(' ')
        cleaned_main_text_seg = ''
        for w in words:
            if len(w) != 0 and w not in stopwords:
                if w[0] not in stop_prefixes:
                    cleaned_main_text_seg += w
                    cleaned_main_text_seg += ' '
        cleaned_main_text_seg = cleaned_main_text_seg.strip()
        paper.main_text_seg = cleaned_main_text_seg
        count += 1
        if count % 500 == 0:
            print('Processed %d Papers!' % count)
    session.commit()
    print('Finished!')


# 生成摘要的分词结果（字段abstract_seg）
def gen_abstract_seg(journal, stopwords=set()):
    print('\nGenerating ABSTRACT_SEG of Journal %s...' % journal)
    session = DBSession()
    papers = session.query(Paper).filter(Paper.journal == journal and
                                         Paper.abstract is not None and
                                         Paper.abstract_seg is None)
    count = 0
    for paper in papers:
        count += 1
        ori_abstract = paper.abstract
        segment_abstract = ' '.join(jieba.cut(ori_abstract, cut_all=False))
        new_abstract = ''
        for w in segment_abstract.split(' '):
            if not (w in stopwords or w == ''):
                new_abstract += w
                new_abstract += ' '
        new_abstract = new_abstract.strip()
        paper.abstract_seg = new_abstract
        if count % 100 == 0:
            print('Processed %d Articles!' % count)
    session.commit()
    print('Finished!')


# 生成关键词表txt（规定最低词长度与最高词长度）
def gen_keyword_file(path, minlen, maxlen, stopwords):
    print('\nGenerating KEYWORD FILE...')
    keywords_file_name = path + '/keywords_' + str(minlen) + '-' + str(maxlen) + '.txt'
    keywords_file = open(keywords_file_name, 'w', encoding='utf-8')
    keywords = set()
    session = DBSession()
    papers = session.query(Paper).all()
    for paper in papers:
        keyword_seg = paper.keyword_seg
        for keyword in keyword_seg.split(' '):
            if minlen <= len(keyword) <= maxlen and keyword[0] not in stopwords:
                keywords.add(keyword)
    count = 0
    for keyword in keywords:
        print(keyword)
        keywords_file.write(keyword)
        keywords_file.write('\n')
        count += 1
    keywords_file.flush()
    keywords_file.close()
    print('Finished! Added %d Keywords!' % count)


# 生成关键词的分词结果 （字段keyword_seg）
# 去掉了带空格的关键词
def gen_keyword_seg(journal):
    session = DBSession()
    papers = session.query(Paper).filter(Paper.journal == journal and
                                         Paper.keyword is not None and
                                         Paper.keyword_seg is None)
    print('\nGenerate KEYWORD_SEG of journal %s...' % journal)
    count = 0
    error_count = 0
    for paper in papers:
        ori_keywords = paper.keyword.replace(';', ' ').replace('；', ' ').replace('：', ' ')
        '''
        keywords = list()
        for keyword in ori_keywords.split('/'):
            contain_space = False
            for s in [' ', '　']:
                if keyword.find(s) != -1:
                    contain_space = True
                    break
            if not contain_space:
                keywords.append(keyword)
        '''
        count += 1
        if len(ori_keywords) <= 1:
            print('Error at paper %d -----> %s, %s, %s' % (paper.id, paper.year, paper.journal, paper.phase))
            error_count += 1
        keyword_seg = ori_keywords
        paper.keyword_seg = keyword_seg
    print('Processed %d articles with %d errors occured!' % (count, error_count))
    session.commit()
    print('Finished!')


# 生成正文的分词结果（字段main_text_seg):
def gen_main_text_seg(journal, stopwords=set()):
    print('\nGenerate MAIN_TEXT_SEG of journal %s...' % journal)
    session = DBSession()
    papers = session.query(Paper).filter(Paper.journal == journal).filter(Paper.main_text is not None). \
        filter(Paper.content_clean is not None).filter(Paper.section is None).\
        filter(Paper.main_text_seg is None).all()
    count = 0
    for paper in papers:
        main_text_new = ''
        lines = paper.content_clean.splitlines()
        if len(lines) >= 3:
            for i in range(len(lines)):
                if not (i == 0 or lines[i].startswith('#')):
                    main_text_new += lines[i].replace('.', '').replace('\n', '').replace('\r', '').strip().\
                        replace('Web20', 'Web2.0')
        if main_text_new == '':
            print('Main Text is NULL at %s -----> %s, %s, %s' % (paper.title, journal, paper.year, paper.issue))
        else:
            main_text_seg = ' '.join(jieba.cut(main_text_new, cut_all=False))
            main_text_seg_new = ''
            for w in main_text_seg.split(' '):
                if w not in stopwords:
                    main_text_seg_new += w
                    main_text_seg_new += ' '
            main_text_seg_new = main_text_seg_new.strip()
            paper.main_text_seg = main_text_seg_new
        count += 1
        if count % 100 == 0:
            print('Processed %d Papers!' % count)
    session.commit()
    print('Finished!')


# 生成正文的分词结果，针对特定的paperid，临时用，一般不用:
def gen_main_text_seg_temp(stopwords=set()):
    print('\nGenerate MAIN_TEXT_SEG (Temporary Method)...')
    paper_ids = {17062, 17063, 17068, 17069, 17071, 17072, 17075, 17076}
    session = DBSession()
    for paper_id in paper_ids:
        papers = session.query(Paper).filter(Paper.id == paper_id).all()
        paper = papers[0]
        print('Processing Paper ID: %d' % paper.id)
        lines = paper.content_clean.splitlines()
        i = 0
        while True:
            if i >= len(lines):
                break
            pos = lines[i].rfind('#') + 1
            line = lines[i][pos:]
            if line.startswith('D'):
                i += 1
                break
            i += 1
        main_text_new = ''
        while i < len(lines):
            pos = lines[i].rfind('#') + 1
            line = lines[i][pos:]
            if not line.startswith('.'):
                main_text_new += line.replace('\r', '').replace('\n', '').strip().replace('Web20', 'Web2.0')
            i += 1
        if main_text_new == '' or main_text_new == ' ':
            print('Main Text is NULL at %s -----> %s, %s, %s' % (paper.title, paper.journal, paper.year, paper.issue))
        else:
            main_text_seg = ' '.join(jieba.cut(main_text_new, cut_all=False))
            main_text_seg_new = ''
            for w in main_text_seg.split(' '):
                if w not in stopwords:
                    main_text_seg_new += w
                    main_text_seg_new += ' '
            main_text_seg_new = main_text_seg_new.strip()
            paper.main_text_seg = main_text_seg_new
        session.commit()
    print('Finished!')


# 生成正文分词结果，补全无section字段的情况（辅助）
def gen_main_text_seg_without_section(journal, stopwords=set()):
    print('\nGenerate MAIN_TEXT_SEG of journal %s...' % journal)
    session = DBSession()
    papers = session.query(Paper).filter(Paper.journal == journal).filter(Paper.year == '2016').all()
    count = 0
    for paper in papers:
        if paper.section is not None:
            continue
        main_text_new = ''
        lines = paper.content_clean.splitlines()
        if len(lines) >= 3:
            for i in range(len(lines)):
                if not (i <= 5 or lines[i].startswith('.')):
                    line = lines[i][5:].strip().\
                        replace('Web20', 'Web2.0')
                    if len(line) > 0:
                        if not line.startswith('万方'):
                            main_text_new += line
        if main_text_new == '' or main_text_new is None:
            print('Main Text is NULL at %s -----> %s, %s, %s' % (paper.title, journal, paper.year, paper.issue))
        else:
            main_text_seg = ' '.join(jieba.cut(main_text_new, cut_all=False))
            main_text_seg_new = ''
            for w in main_text_seg.split(' '):
                if w not in stopwords:
                    main_text_seg_new += w
                    main_text_seg_new += ' '
            main_text_seg_new = main_text_seg_new.strip()
            paper.main_text_seg = main_text_seg_new
        count += 1
        print('Processed Paper ID %d...' % paper.id)
    print('Processed %d Papers!' % count)
    session.commit()
    print('Finished!')


# 生成标题的分词结果（字段title_seg）
def gen_title_seg(journal, stopwords=set()):
    print('\nGenerate TITLE_SEG of journal %s...' % journal)
    session = DBSession()
    papers = session.query(Paper).filter(Paper.journal == journal and
                                         Paper.title is not None and
                                         Paper.title_seg is None)
    count = 0
    for paper in papers:
        title = paper.title.strip().replace('\n', '').replace('\r', '').replace(' ', '').replace('　', '')
        segment_title = ' '.join(jieba.cut(title, cut_all=False))
        new_title = ''
        for w in segment_title.split(' '):
            if not (w in stopwords or w == ''):
                new_title += w
                new_title += ' '
        new_title = new_title.strip()
        paper.title_seg = new_title
        count += 1
        if count % 100 == 0:
            print('Processed %d Articles!' % count)
    session.commit()
    print('Finished!')


# 插入论文标签
def insert_label(label_file_path=''):
    print('\nInserting LABEL of the papers')
    label_dict = get_label_dict_lda(label_file_path)
    session = DBSession()
    papers = session.query(Paper).all()
    count = 0
    for paper in papers:
        paper.label_id = label_dict[paper.id]
        count += 1
        if count % 500 == 0:
            print('%d\t%d' % (paper.id, label_dict[paper.id]))
            print('Processed %d Articles!' % count)
    session.commit()
    print('Finished!')


# 生成jieba分词词典
def load_dict(dict_path=''):
    print('\nLoading Dictionary...')
    print('Jieba Dictionary Path: %s' % dict_path)
    if dict_path != '':
        jieba.load_userdict(dict_path)
        with open(dict_path, encoding='utf-8') as d:
            print('The Dictionary Contains %d Words!' % len(d.readlines()))
            d.close()
    print('Finished!')


# 生成停用词表
def load_stopwords(stopwords_path=''):
    print('\nLoading Stopwords...')
    print('Stopwords Path: %s' % stopwords_path)
    stopwords = set()
    if stopwords_path != '':
        with open(stopwords_path, encoding='utf-8') as swf:
            for sw in swf.readlines():
                sw = sw.replace('\n', '').replace('\r', '')
                stopwords.add(sw)
            swf.close()
    print('Number of Stopwords: %d' % len(stopwords))
    print('Finished!')
    return stopwords

# main函数
def main():
    os.chdir('E:\HDQ\Projects\Graduation')

    # insert_label('E:\HDQ\Projects\Graduation\\output\\doc_topic-15_1124.txt')
    output_paper_texts('./output/texts_without_main_text', with_main_text=False)
    # test_output_content_clean(load_dict('./res/dictionary/Dictionary.txt')
    # gen_main_text_seg_temp(stopwords)
    # stop_main_text = load_stopwords('./res/stopwords/main_text_stop.txt')
    # stopwords = load_stopwords('./res/stopwords/stopwords_17_10_30.txt')
    # stop_prefixes = load_stopwords('./res/stopwords/stop_prefixes.txt')
    # load_dict('./res/dictionary/Dictionary.txt')
    # stopwords_title = load_stopwords('./res/stopwords/stopwords_17_11_9.txt')
    # stopwords_abstract = load_stopwords('./res/stopwords/stopwords_17_11_9.txt')
    # journals = [
    #     '情报科学', '情报理论与实践', '情报学报', '情报杂志', '情报资料工作',
    #     '图书情报工作', '图书情报知识', '图书与情报', '现代情报', '现代图书情报技术'
    # ]
    # for journal in journals:
    #     clean_main_text_seg_deep(journal, stopwords=stopwords, stop_prefixes=stop_prefixes)
    #   clean_main_text_seg(journal, stop_prefixes)
    #   gen_abstract_seg(journal, stopwords=stopwords_abstract)
    #   gen_title_seg(journal, stopwords=stopwords_title)

    # test_output_main_text_seg()
    # clean_abstract()
    # output_paper_texts('E:\HDQ\Resources\\texts_without_main_text_11_9')
    #     gen_main_text_seg_without_section(journal, stopwords=stopwords)
    # gen_main_text_seg(journal, stopwords=stop_prefixes)


# 对论文标题、摘要、关键词、正文生成txt文件集
def output_paper_texts(path='./', with_main_text=False):
    print('\nOutputting Papers to Texts...')
    session = DBSession()
    papers = session.query(Paper).all()
    count = 0
    for paper in papers:
        paper_id = str(paper.id)
        with open(path + '/' + paper_id + '.txt', 'w', encoding='utf-8') as f:
            f.write(paper.title_seg)
            f.write('\n')
            f.write(paper.abstract_seg)
            f.write('\n')
            keywords = list()
            for kw in paper.keyword_seg.split(' '):
                if kw not in {''}:
                    keywords.append(kw)
            f.write(' '.join(keywords))
            f.write('\n')
            if with_main_text:
                f.write(paper.main_text_seg)
                f.write('\n')
            f.flush()
            f.close()
        count += 1
        if count % 500 == 0:
            print('Processed %d papers...' % count)
    print('Finished!')


# 测试：打印content_clean
def test_output_content_clean():
    print('\nTesting outputting CONTENT_CLEAN field...\n')
    session = DBSession()
    papers = session.query(Paper).filter(Paper.id == 8995)
    for i in range(1):
        for line in papers[i].content_clean.splitlines():
            print(line)
        print()


# 测试：打印main_text_seg
def test_output_main_text_seg():
    print('\nTesting outputting MAIN_TEXT_SEG field...\n')
    session = DBSession()
    papers = session.query(Paper).filter(Paper.id == 8995)
    for i in range(1):
        print(papers[i].filename, papers[i].journal, papers[i].year, papers[i].issue)
        print(papers[i].main_text_seg)


if __name__ == '__main__':
    main()
