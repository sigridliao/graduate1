'''
content的sql字段编码需为utf8mb4
sql字段类型text长度如下：
TINYTEXT	256 bytes
TEXT	65,535 bytes	~64kb
MEDIUMTEXT	 16,777,215 bytes	~16MB
LONGTEXT	4,294,967,295 bytes	~4GB
有些论文txt长度大于64kb，故需要MEDIUMTEXT
'''
from sqlalchemy import Column, String, TEXT, INTEGER, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import re
import jieba
import math
import time
import jieba.posseg

# 创建对象的基类:
BaseModel = declarative_base()  # 建了一个 BaseModel 类，这个类的子类可以自动与一个表关联。


# 定义paper对象:
class Paper(BaseModel):
    # 表的名字:
    # __tablename__ = 'paper_test'
    # __tablename__ = 'paper_title_abstract'
    __tablename__ = 'paper_clean'

    # 表的结构:
    id = Column(INTEGER(), primary_key=True)
    filename = Column(String(255))
    journal = Column(String(255))
    year = Column(String(4))
    phase = Column(String(2))
    content = Column(TEXT(255))

    # paper_copy 增加的
    zhengwen = Column(TEXT(255))
    title = Column(String(255))
    frontcontent = Column(TEXT(255))
    abstract = Column(TEXT(255))
    keyword = Column(String(255))
    sectiontitle = Column(TEXT(255))
    cleancontent = Column(TEXT(255))
    # keywordfen = Column(String(255))
    # firstpara = Column(TEXT(255))
    # middlepara = Column(TEXT(255))
    # lastpara = Column(TEXT(255))
    # abstractfen = Column(TEXT(255))
    # titlefen = Column(String(255))
    # sectionfen = Column(String(1000))
    # wordcount = Column(TEXT(255))
    # wordsum = Column(INTEGER())


# 初始化数据库连接:
# echo为True是调试模式，会打印信息
# 必须设置为utf8mb4
engine = create_engine('mysql+mysqlconnector://root:123456@localhost:3306/paper?charset=utf8mb4', echo=False)
# 创建DBSession类型:
DBSession = sessionmaker(bind=engine)


# jieba.load_userdict("F:\\MyPaper\\词典\\new\\keywords_3-6.txt")
# with open('F:\\MyPaper\\词典\\stopwords_combine.txt', 'r', encoding='utf-8') as f:
#     stopwordset = set([stopword for stopword in f.read().split('\n')])

def insert(paper):
    '''
    循环将论文列表中的论文插入数据库
    :param papers:
    :return:
    '''
    session = DBSession()  # 创建session对象
    session.add(paper)  # 循环添加到session
    session.commit()  # 提交即保存到数据库:
    session.close()  # 关闭session:


def insertbatch(papers):
    '''
    循环将论文列表中的论文插入数据库
    :param papers:
    :return:
    '''
    session = DBSession()  # 创建session对象
    for paper in papers:
        session.add(paper)  # 循环添加到session
    session.commit()  # 提交即保存到数据库:
    session.close()  # 关闭session:


def get_paper_by_id(paperid):
    session = DBSession()  # 创建Session:
    # 创建Query查询，filter是where条件，最后调用one()返回唯一行，如果调用all()则返回所有行:
    paper = session.query(Paper).filter(Paper.id == paperid).one()
    # 打印类型和对象的name属性:
    print('type:', type(paper))
    print('filename:', paper.content)
    # 关闭Session:
    session.close()


def get_papers_by_journal(journal):
    session = DBSession()  # 创建Session:
    # 创建Query查询，filter是where条件，最后调用one()返回唯一行，如果调用all()则返回所有行:
    papers = session.query(Paper).filter(Paper.journal == journal).all()
    return papers


def get_papers_by_journal_and_update_frontcontent(journal, frontcontent):
    pass

def normalization_l2(list_vector):  # 返回的是list
    if type(list_vector) == str:
        list_vector = eval('[' + list_vector + ']')
    list_square = list(map(lambda x: x ** 2, list_vector))
    square_sum_sqrt = math.sqrt(sum(list_square))
    if square_sum_sqrt == 0: return list_vector
    result = list(map(lambda x: x / square_sum_sqrt, list_vector))
    return result


def normalization_l1(list_vector):  # 返回的是list
    if type(list_vector) == str:
        list_vector = eval('[' + list_vector + ']')
    total = sum(list_vector)
    if total == 0: return list_vector
    result = [i / total for i in list_vector]
    return result


def normalization_l2_by_file():
    with open(r'F:\MyPaper\Word2vec\200维_10\TF_IDF.txt', 'r', encoding='utf8') as f:
        content = f.read().splitlines()
    newlist = list(map(normalization_l2, content))
    with open(r'F:\MyPaper\Word2vec\200维_10\TF_IDF_l2.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join([str(i).strip('[]').replace(' ', '') for i in newlist]))


def normalization_l1_by_file():
    with open(r'F:\MyPaper\Word2vec\200维_500\TF_IDF.txt', 'r', encoding='utf8') as f:
        content = f.read().splitlines()
    newlist = list(map(normalization_l1, content))
    with open(r'F:\MyPaper\Word2vec\200维_500\TF_IDF_l1.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join([str(i).strip('[]').replace(' ', '') for i in newlist]))


def cal_word_sum():
    session = DBSession()
    papers = session.query(Paper).all()
    for paper in papers:
        if not paper.id % 50:
            print(paper.id)
        word_sum = 0
        wordcountdict = eval(paper.wordcount)
        for k in wordcountdict:
            word_sum += wordcountdict[k][0]
        paper.wordsum = word_sum
    session.commit()


def gen_paperid_list():
    paper_id_list = []
    session = DBSession()
    papers = session.query(Paper).all()
    for paper in papers:
        paper_id_list.append(str(paper.id))
    with open(r'F:\MyPaper\词典\paperid_list.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(paper_id_list))


def cal_TFIDF_word2vec():  # 计算通过word2vec聚类后的特征的TFIDF
    with open(r'F:\MyPaper\Word2vec\200维_10\class.sorted.txt', 'r', encoding='utf8') as f:
        content = [(line.split(' ')[0], int(line.split(' ')[1])) for line in f.read().splitlines()]
    with open(r'F:\MyPaper\词典\new\total_word_DF_sort.txt', 'r', encoding='utf8') as f:
        DF_dict = dict([(line.split(' ')[0], int(line.split(' ')[1])) for line in f.read().splitlines()])
    worddict = {}
    for k, v in content:
        if not v in worddict: worddict[v] = set()
        worddict[v].add(k)
    print(len(worddict))
    session = DBSession()
    papers = session.query(Paper).all()
    total_TFIDF = []
    for paper in papers:
        TF = []
        if not paper.id % 500:
            print(paper.id)
        wordcountdict = eval(paper.wordcount)
        worddict_key = list(worddict.keys())  # 让距离特征有顺序
        worddict_key.sort()
        for i in worddict_key:
            TF_item = []
            for word in worddict[i]:
                if word in wordcountdict:
                    if word not in DF_dict:
                        DF_dict[word] = 1
                    idf = math.log((11947 / DF_dict[word]) + 0.01)
                    TF_item.append(int(wordcountdict[word][0]) * idf)
            TF.append(sum(TF_item))
        # TF = normalization_l2(TF)
        total_TFIDF.append(','.join(map(lambda x: str(x), TF)))
    with open(r'F:\MyPaper\Word2vec\200维_100\TF_IDF.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(total_TFIDF))


def cal_TF_word2vec():  # 计算通过word2vec聚类后的特征的TF
    with open(r'F:\MyPaper\Word2vec\200维_25\class.sorted.txt', 'r', encoding='utf8') as f:
        content = [(line.split(' ')[0], line.split(' ')[1]) for line in f.read().splitlines()]
    worddict = {}
    for k, v in content:
        if not v in worddict: worddict[v] = set()
        worddict[v].add(k)
    print(len(worddict))
    session = DBSession()
    papers = session.query(Paper).all()
    total_TFIDF = []
    for paper in papers:
        TF = []
        if not paper.id % 500:
            print(paper.id)
        wordcountdict = eval(paper.wordcount1)
        for i in list(map(lambda x: int(x), worddict.keys())):
            TF_item = []
            for word in worddict[str(i)]:
                if word in wordcountdict:
                    TF_item.append(int(wordcountdict[word][0]))
            TF.append(sum(TF_item))
        total_TFIDF.append(','.join(map(lambda x: str(x), TF)))
    with open(r'F:\MyPaper\Word2vec\200维_25\new\TF.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(total_TFIDF))


def gen_corpus_file():
    session = DBSession()
    papers = session.query(Paper).all()
    content = []
    f = open('F:\\MyPaper\\corpus\\train.txt', 'w', encoding='utf-8')
    for paper in papers:
        title = paper.titlefen.replace('/', ' ')
        keyword = paper.keywordfen.replace('/', ' ') if paper.keywordfen else ''
        abstract = paper.abstractfen.replace('/', ' ') if paper.abstractfen else ''
        section = paper.sectionfen.replace('/', ' ') if paper.sectionfen else ''
        firstpara = paper.firstpara.replace('/', ' ') if paper.firstpara else ''
        middlepara = paper.middlepara.replace('/', ' ') if paper.middlepara else ''
        lastpara = paper.lastpara.replace('/', ' ') if paper.lastpara else ''

        content.append(title)
        content.append(keyword)
        content.append(abstract)
        content.append(section)
        content.append(firstpara)
        content.append(middlepara)
        content.append(lastpara)

        if not paper.id % 200:
            f.write('\n'.join(content))
            content = []
            print(paper.id)

    f.write('\n'.join(content))
    f.close()
    session.close()


def cal_TFIDF2():
    with open('F:\\MyPaper\\词典\\new\\total_word_DF_100-999.txt', 'r', encoding='utf-8') as f:
        wordlist = f.read().splitlines()
    word_sum = len(wordlist)
    wordlist_tuple = [tuple(word.split(' ')) for word in wordlist]
    session = DBSession()
    papers = session.query(Paper).all()
    total_TFIDF = []
    print('加载文件完成')
    for paper in papers:
        TF_IDF_list = []
        total = 0
        if not paper.id % 500:
            print(paper.id)
        countdict = eval(paper.wordcount)
        print('------------开始第id为 %s 的论文权值计算--------------' % paper.id)
        starttime = time.time()
        for num, (word, DF) in enumerate(wordlist_tuple):
            if word in countdict:
                TF = countdict[word][0]
                IDF = math.log((11947 / int(DF)) + 0.01)
                TF_IDF = TF * IDF
                TF_IDF_list.append([num, TF * IDF])
                total += (TF_IDF ** 2)
        total = math.sqrt(total)
        # print(TF_IDF_list)
        # print(total)
        TF_IDF_list = list(map(lambda x: [x[0], x[1] / total], TF_IDF_list))
        # print(TF_IDF_list)

        TF_IDF_list1 = [0] * word_sum
        for x in TF_IDF_list:
            TF_IDF_list1[x[0]] = x[1]
        TF_IDF_list1_str = str(TF_IDF_list1).strip('[]')
        total_TFIDF.append(TF_IDF_list1_str)
        print('------------计算完毕，花费时间为%s-----------------' % (time.time() - starttime))

    with open('F:\\MyPaper\\词典\\new\\TF_IDF_100-999.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(total_TFIDF))


def cal_TFIDF():
    with open('F:\\MyPaper\\词典\\new\\total_word_DF_100-1000.txt', 'r', encoding='utf-8') as f:
        wordlist = f.read().splitlines()
    session = DBSession()
    papers = session.query(Paper).all()
    total_TFIDF = []
    print('加载文件完成')
    for paper in papers:
        TF_IDF = []
        if not paper.id % 500:
            print(paper.id)
        for word_DF in [tuple(word.split(' ')) for word in wordlist]:
            TF = (eval(paper.wordcount).get(word_DF[0], [0]))[0]
            DF = int(word_DF[1])
            tfidf = TF * math.log((11947 / DF) + 0.01)
            TF_IDF.append(tfidf)
        total = math.sqrt(sum(map(lambda x: x ** 2, TF_IDF)))
        print(total)
        TF_IDF = list(map(lambda x: x / total, TF_IDF))
        print(TF_IDF)
        TF_IDF = str(TF_IDF).strip('[]')
        total_TFIDF.append(TF_IDF)

    with open('F:\\MyPaper\\词典\\new\\TF_IDF_100-200_clean.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(total_TFIDF))


def total_word_clean():
    with open('F:\\MyPaper\\词典\\total_word.txt', 'r', encoding='utf-8') as f:
        wordlist = f.read().splitlines()
    p = re.compile(r'^[0-9a-zA-Z\.\(\)&]+$')

    print(len(wordlist))
    with open('F:\\MyPaper\\词典\\total_word_clean.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join([word for word in wordlist if not p.match(word) and len(word) > 1 and len(word) < 11]))


def filter_word_remove_English():
    with open('F:\\MyPaper\\词典\\new\\total_word_DF_100-200.txt', 'r', encoding='utf-8') as f:
        wordlist = f.read().splitlines()
    with open('F:\\MyPaper\\词典\\new\\total_word_DF_100-200_clean.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join([word for word in wordlist if not hasalpha(word.split(' ')[0])]))


def select_by_DF():  # 设置DF的阈值，筛选DF大于x小于y的词
    with open('F:\\MyPaper\\词典\\new\\total_word_DF_sort.txt', 'r', encoding='utf-8') as f:
        wordlist = f.read().splitlines()
    with open('F:\\MyPaper\\词典\\new\\total_word_DF_100-999.txt', 'w', encoding='utf-8') as f:
        f.write(
            '\n'.join([word for word in wordlist if int(word.split(" ")[1]) >= 100 and int(word.split(" ")[1]) <= 999]))


def sort_TForDF_bylen():
    with open('F:\\MyPaper\\词典\\new\\total_word_TF_norm.txt', 'r', encoding='utf-8') as f:
        wordlist = f.read().splitlines()
    # wordlist.sort(key = lambda x:int(x.split(' ')[1]),reverse=True)
    wordlist.sort(key=lambda x: len(x.split(' ')[0]))
    with open('F:\\MyPaper\\词典\\new\\total_word_TF_norm_sort.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(wordlist))


def cal_DF2():
    with open('F:\\MyPaper\\词典\\new\\total_word.txt', 'r', encoding='utf-8') as f:
        wordlist = f.read().splitlines()
    DF_dict = dict([(word, 0) for word in wordlist])
    session = DBSession()
    papers = session.query(Paper).all()
    for paper in papers:
        if not paper.id % 500:
            print(paper.id)
        for word in eval(paper.wordcount).keys():
            if word in DF_dict:
                DF_dict[word] += 1
    result = [k + ' ' + str(DF_dict[k]) for k in DF_dict.keys()]
    result.sort(key=lambda x: int(x.split(' ')[1]))
    with open('F:\\MyPaper\\词典\\new\\total_word_DF1.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(result))


def cal_TFxIDF():
    with open('F:\\MyPaper\\词典\\new\\total_word_DF_sort.txt', 'r', encoding='utf-8') as f:
        DF_list = f.read().splitlines()
    with open('F:\\MyPaper\\词典\\new\\total_word_TF_norm_sort.txt', 'r', encoding='utf-8') as f:
        TF_list = f.read().splitlines()
    TFxIDF = [0] * len(DF_list)
    for (num, word) in enumerate(DF_list):
        TFxIDF[num] = (float(TF_list[num].split(' ')[1])) * (math.log(11947 / (int(word.split(' ')[1]) + 0.01)))
        # print(TFxIDF[num])
        TFxIDF[num] = word.split(' ')[0] + ' ' + str(TFxIDF[num])
    TFxIDF.sort(key=lambda x: float(x.split(' ')[1]), reverse=True)
    with open('F:\\MyPaper\\词典\\new\\total_word_norm_TFxIDF.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(TFxIDF))


def cal_TF():
    with open('F:\\MyPaper\\词典\\new\\total_word.txt', 'r', encoding='utf-8') as f:
        wordlist = f.read().splitlines()
    TF_dict = dict([(word, 0) for word in wordlist])
    session = DBSession()  # 创建Session:
    papers = session.query(Paper).all()
    for paper in papers:
        if not paper.id % 500:
            print(paper.id)
        countdict = eval(paper.wordcount)
        total_word = 0
        for word in countdict.keys():
            total_word += countdict[word][0]
        for word in TF_dict.keys():
            if word in countdict:
                TF_dict[word] += (countdict[word][0] / total_word)
    result = []
    for word in TF_dict.keys():
        result.append((word + ' ' + str(TF_dict[word])))
    result.sort(key=lambda x: float(x.split(' ')[1]), reverse=True)
    with open('F:\\MyPaper\\词典\\new\\total_word_TF_norm.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(result))


def cal_DF():
    with open('F:\\MyPaper\\词典\\new\\total_word.txt', 'r', encoding='utf-8') as f:
        wordlist = f.read().splitlines()
    wordDF = [0] * len(wordlist)
    session = DBSession()  # 创建Session:
    papers = session.query(Paper).all()
    for num, word in enumerate(wordlist):
        for paper in papers:
            if word in eval(paper.wordcount).keys():
                wordDF[num] += 1
    result = []
    for num, word in enumerate(wordlist):
        result.append(word + ' ' + wordDF[num])
    with open('F:\\MyPaper\\词典\\total_word_DF.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(result))


def gen_wordlist_file():
    session = DBSession()  # 创建Session:
    papers = session.query(Paper).all()
    total_word_set = set()
    for paper in papers:
        wordcountdict = eval(paper.wordcount)
        total_word_set = set(wordcountdict.keys()) | total_word_set
    with open('F:\\MyPaper\\词典\\new\\total_word.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(list(total_word_set), key=len)))
    session.close()


def wordcount():
    session = DBSession()  # 创建Session:
    papers = session.query(Paper).all()
    for paper in papers:
        wordcountdict = {}
        if paper.titlefen:
            for word in paper.titlefen.split('/'):
                wordcountdict.setdefault(word, [0] * 5)[1] += 1
        if paper.abstractfen:
            for word in paper.abstractfen.split('/'):
                wordcountdict.setdefault(word, [0] * 5)[2] += 1
        if paper.keywordfen:
            for word in paper.keywordfen.split('/'):
                wordcountdict.setdefault(word, [0] * 5)[3] += 1
        if paper.sectionfen:
            for word in paper.sectionfen.split('/'):
                wordcountdict.setdefault(word, [0] * 5)[4] += 1
        # if paper.firstpara:
        #     for word in paper.firstpara.split('/'):
        #         wordcountdict.setdefault(word, [0] * 8)[5] += 1
        # if paper.lastpara:
        #     for word in paper.lastpara.split('/'):
        #         wordcountdict.setdefault(word, [0] * 8)[6] += 1
        # if paper.middlepara:
        #     for word in paper.middlepara.split('/'):
        #         wordcountdict.setdefault(word, [0] * 8)[7] += 1
        for key in wordcountdict:
            wordcountdict[key][0] = sum(wordcountdict[key][1:])
        paper.wordcount1 = str(wordcountdict)
    session.commit()


def split_content():
    session = DBSession()  # 创建Session:
    papers = session.query(Paper).all()
    for paper in papers:
        content = paper.cleancontent.splitlines()
        sectionnum = []
        for num, line in enumerate(content):
            if line.startswith('###'):
                sectionnum.append(num)
        paper.firstpara = content[sectionnum[0] + 1]
        if sectionnum[-1] <= len(content) - 2:
            paper.lastpara = content[sectionnum[-1] + 1]
        else:
            paper.lastpara = None
        middleparalist = content[sectionnum[0] + 2:sectionnum[-1]]
        paper.middlepara = '.'.join([line for line in middleparalist if not line.startswith('###')])
    session.commit()


def fenci_by_column():
    session = DBSession()  # 创建Session:
    papers = session.query(Paper).filter(Paper.journal != '情报科学')
    for paper in papers:
        if not paper.id % 20:
            print(paper.id)
        if paper.firstpara != None:
            paper.firstpara = fenci_by_pos(paper.firstpara.replace('\n', ''))
        if paper.middlepara != None:
            paper.middlepara = fenci_by_pos(paper.middlepara.replace('\n', ''))
        if paper.lastpara != None:
            paper.lastpara = fenci_by_pos(paper.lastpara.replace('\n', ''))
        paper.abstractfen = fenci_by_pos(paper.abstract.replace('\n', ''))
        paper.titlefen = fenci_by_pos(paper.title.replace('\n', ''))
        paper.sectionfen = fenci_by_pos(paper.sectiontitle.replace('\n', ''))
    session.commit()


def filter_by_wordlen():
    with open('F:\\MyPaper\\词典\\new\\keywords.txt', 'r', encoding='utf-8') as f:
        wordlist = f.read().splitlines()
        wordlist = [word for word in wordlist if len(word) >= 3 and len(word) <= 6 and not hasalpha(word)]
    with open('F:\\MyPaper\\词典\\new\\keywords_3-6.txt', 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(wordlist))


def addfreq():
    with open('F:\\MyPaper\\词典\\keywords_clean.txt', 'r', encoding='utf-8') as f:
        wordlist = [word.rstrip() + ' 100\n' for word in f.readlines()]
    with open('F:\\MyPaper\\词典\\keywords_clean1.txt', 'w', encoding='utf-8') as f:
        f.writelines(wordlist)


def fenci(string):
    global stopwordset
    fencilist = [word for word in jieba.cut(string) if word not in stopwordset and not word.isdigit()]
    return '/'.join(fencilist)


def fenci_by_pos(string):
    global stopwordset
    fencilist = [word.word for word in jieba.posseg.cut(string) if
                 word.word not in stopwordset and not word.word.isdigit() and \
                 word.flag not in 'rmdueywfqpct' and word.flag != 'nr' and len(word.word) > 1]
    return '/'.join(fencilist)


def combine_stopword():
    with open('F:\\MyPaper\\词典\\stopwords2612.txt', 'r', encoding='utf-8') as f:
        wordlist = f.readlines()
    with open('F:\\MyPaper\\词典\\stopwords1893.txt', 'r', encoding='utf-8') as f:
        wordlist.extend(f.readlines())
    stopwordlist = list(set(wordlist)).sort(key=len)
    with open('F:\\MyPaper\\词典\\stopwords_combine.txt', 'w', encoding='utf-8') as f:
        f.writelines(stopwordlist)


def collect_keyword():
    session = DBSession()  # 创建Session:
    papers = session.query(Paper).filter(Paper.keywordfen != None)
    keywordset = set()
    for paper in papers:
        keywordset = keywordset | set(paper.keywordfen.split('/'))
    keywordlist = list(keywordset)
    keywordlist.sort(key=len)
    keywordlist.remove('')
    with open('词典\\new\\keywords.txt', 'w', encoding='utf-8') as f:
        f.writelines([line + '\n' for line in keywordlist])
    session.close()


def clean_keyword(journal):
    session = DBSession()  # 创建Session:
    papers = session.query(Paper).filter(Paper.journal == journal).filter(Paper.keywordfen != None)
    for paper in papers:
        # if ',' in paper.keywordfen or '//' in paper.keywordfen or '，' in paper.keywordfen:
        #     paper.keywordfen = paper.keywordfen.replace('，','/').replace(',','/').replace('//','/')
        keywordfen = paper.keyword.replace(';', '/').replace('；', '/').replace('：', '/').replace(' ', '/')
        if keywordfen == paper.keyword:
            #     break
            paper.keywordfen = keywordfen
    session.commit()


def exact_linenum_and_text(string):  # 输出元组（行号，内容）
    regex = r'^(.*)#(\d+?)#$'
    try:
        result = re.search(regex, string)
    except:
        print('错误匹配', string)
        raise Exception
        # return string
    return (int(result.group(2)), result.group(1))


def gen_cleancontent(journal):
    session = DBSession()  # 创建Session:
    papers = session.query(Paper).filter(Paper.journal == journal).filter(Paper.zhengwen != None) \
        .filter(Paper.year == '2016')
    for paper in papers:
        flag = True
        contentdict = dict(map(exact_linenum_and_text, paper.zhengwen.splitlines()))
        sectiontitlelist = map(exact_linenum_and_text, paper.sectiontitle.splitlines())
        for num, sectiontitle in enumerate(sectiontitlelist):
            if sectiontitle[0] in contentdict:
                print(paper.id, paper.journal, sectiontitle)
                # flag = False
            contentdict[sectiontitle[0]] = '\n###' + str(num) + '#' + sectiontitle[1] + '\n'
        keys = list(contentdict.keys())
        keys.sort()
        cleancontent = map(contentdict.get, keys)
        if flag:
            paper.cleancontent = '.'.join(cleancontent)
    session.commit()


def exact_zhengwen(journal):
    session = DBSession()  # 创建Session:
    papers = session.query(Paper).filter(Paper.abstract != None).filter(Paper.journal == journal) \
        .filter(Paper.zhengwen == None).filter(Paper.year == '2016')
    errrorcount = 0

    font_size_dict_2016 = {
        '情报科学': ['8.9999###', '8.99985###'],
        '情报理论与实践': ['8.8302###', '8.24798###', '8.83019###'],
        '情报学报': ['8###', '9###', '10###', '5###', '6###', '7###', '3###'],
        '情报杂志': ['8.83019###', '9.7035###'],
        '情报资料工作': ['8.22145###', '9.4124###', '8.47059###', '7.72318###', '8.71972###', '4.48443###'],
        '图书情报工作': ['9.7035###', '9.12129###'],
        '图书情报知识': ['8.92722###', '8.9272###'],
        '图书与情报': ['9.41###', '9###'],
        '现代情报': ['8###', '9###'],
        '现代图书情报技术': ['10.02###']
        # 打头的字号
        # 如果识别这个字号打头
        # 就把这一行加进正文
    }

    for paper in papers:
        content = []
        contentlist = paper.content.splitlines()
        for num, line in enumerate(contentlist):
            for font_size in font_size_dict_2016[journal]:
                if line.startswith(font_size):
                    content.append(exacttext(line) + '#' + str(num) + '#')
                    break
        if content == [] or len(content) < 50:
            errrorcount += 1
            paper.zhengwen = None
        else:
            paper.zhengwen = '\n'.join(content)
    print(journal + ' errorcount', errrorcount)
    session.commit()


def exact_sectiontitle(journal):
    session = DBSession()  # 创建Session:
    errorcount = 0
    papers = session.query(Paper).filter(Paper.abstract != None).filter(Paper.journal == journal).filter(
        Paper.sectiontitle == None)
    sectiontitle_dict_2016 = {
        '情报科学': ['10.9999###'],
        '情报理论与实践': ['10.2857###', '9.7035###'],
        '情报学报': ['13###', '12###'],
        '情报杂志': ['10.2857###'],
        # '情报资料工作': ['7.97232###', '9.4124###'],
        '图书情报工作': ['12.7116###', '9.12129###'],
        '图书情报知识': ['8.92722###', '9.7035###'],
        '图书与情报': ['9.99###', '10.96###', '9###', '10###'],
        '现代情报': ['9###', '10###', '10.5###'],
        '现代图书情报技术': ['12###']
    }
    for paper in papers:
        sectiontitlelist = []
        for num, line in enumerate(paper.content.splitlines()):
            for prefix in sectiontitle_dict_2016[journal]:
                if line.startswith(prefix):
                    # or exacttext(line).startswith('0 引') or exacttext(line).startswith('０ 引'):
                    sectiontitlelist.append(exacttext(line) + '#' + str(num) + '#')
                    break
        if sectiontitlelist == [] or len(sectiontitlelist) < 2:
            errorcount += 1
            # paper.sectiontitle = None
        else:
            paper.sectiontitle = '\n'.join(sectiontitlelist)
            # print(sectiontitlelist)
    session.commit()
    print(journal + ': %d errors' % errorcount)


def exact_sectiontitle_qbzlgz():  # 情报资料工作专属 操他妈 段标都在正文中，先提取正文
    session = DBSession()  # 创建Session:
    errorcount = 0
    papers = session.query(Paper).filter(Paper.abstract != None).filter(Paper.journal == '情报资料工作') \
        .filter(Paper.sectiontitle == None).filter(Paper.zhengwen != None).filter(Paper.year == '2016')
    for paper in papers:
        sectiontitlelist = []
        for num, line in enumerate(paper.content.splitlines()):
            line = exacttext(line)
            if line.startswith('1 ') or line.startswith('2 ') or line.startswith('3 ') or line.startswith('4 ') \
                    or line.startswith('5 ') or line.startswith('6 ') or line.startswith('7 ') or line.startswith('0 ') \
                    or line.startswith('１ ') or line.startswith('２ ') or line.startswith('３ ') or line.startswith('０ ') \
                    or line.startswith('４ ') or line.startswith('５ ') or line.startswith('６ ') or line.startswith('７ '):
                sectiontitlelist.append(line + '#' + str(num) + '#')
        if sectiontitlelist == [] or len(sectiontitlelist) < 2:
            errorcount += 1
            # paper.sectiontitle = None
        else:
            paper.sectiontitle = '\n'.join(sectiontitlelist)
            print(sectiontitlelist)
    session.commit()
    print(errorcount)


def clean_sectiontitle2(journal):  # 以序号后无空格来过滤
    session = DBSession()
    errorcount = 0
    papers = session.query(Paper).filter(Paper.abstract != None).filter(Paper.journal == journal) \
        .filter(Paper.sectiontitle != None)
    for paper in papers:
        sectiontitlelist = paper.sectiontitle.splitlines()
        tmp = []
        for num, line in enumerate(sectiontitlelist):
            if line.startswith('1') or line.startswith('2') or line.startswith('3') or line.startswith('4') \
                    or line.startswith('5') or line.startswith('6') or line.startswith('7') or line.startswith('0') \
                    or line.startswith('１') or line.startswith('２') or line.startswith('３') or line.startswith('０') \
                    or line.startswith('４') or line.startswith('５') or line.startswith('６') or line.startswith('７'):
                if num != len(sectiontitlelist) - 1 and exactline(sectiontitlelist[num + 1]) == (exactline(line) + 1) \
                        and not sectiontitlelist[num + 1][0].isdigit():  # 过滤子标题
                    line = exacttext_without_line(line) + sectiontitlelist[num + 1]
                tmp.append(line)
        if tmp == []:
            errorcount += 1
            paper.sectiontitle = None
        else:
            paper.sectiontitle = '\n'.join(tmp)
    print(journal + ': %d errors' % errorcount)
    session.commit()


def clean_sectiontitle3(journal):  # 以序号后有空格来过滤
    session = DBSession()
    errorcount = 0
    papers = session.query(Paper).filter(Paper.abstract != None).filter(Paper.journal == journal) \
        .filter(Paper.sectiontitle != None)
    for paper in papers:
        sectiontitlelist = paper.sectiontitle.splitlines()
        tmp = []
        for num, line in enumerate(sectiontitlelist):
            if line.startswith('1 ') or line.startswith('2 ') or line.startswith('3 ') or line.startswith('4 ') \
                    or line.startswith('5 ') or line.startswith('6 ') or line.startswith('7 ') or line.startswith('0 ') \
                    or line.startswith('１ ') or line.startswith('２ ') or line.startswith('３ ') or line.startswith('０ ') \
                    or line.startswith('４ ') or line.startswith('５ ') or line.startswith('６ ') or line.startswith('７ '):
                if num != len(sectiontitlelist) - 1 and exactline(sectiontitlelist[num + 1]) == (
                    exactline(line) + 1) and not \
                        sectiontitlelist[num + 1][0].isdigit():  # 过滤子标题
                    line = exacttext_without_line(line) + sectiontitlelist[num + 1]
                tmp.append(line)
        if tmp == []:
            errorcount += 1
            paper.sectiontitle = None
        else:
            paper.sectiontitle = '\n'.join(tmp)
    print(errorcount)
    session.commit()


def clean_sectiontitle4(journal):  # 以序号后有空格来过滤，增加大写字 一 二
    session = DBSession()
    errorcount = 0
    papers = session.query(Paper).filter(Paper.abstract != None).filter(Paper.journal == journal) \
        .filter(Paper.sectiontitle != None).filter(Paper.year == '2016')
    for paper in papers:
        sectiontitlelist = paper.sectiontitle.splitlines()
        tmp = []
        for num, line in enumerate(sectiontitlelist):
            if line.startswith('1 ') or line.startswith('2 ') or line.startswith('3 ') or line.startswith('4 ') \
                    or line.startswith('5 ') or line.startswith('6 ') or line.startswith('7 ') or line.startswith('0 ') \
                    or line.startswith('１ ') or line.startswith('２ ') or line.startswith('３ ') or line.startswith('０ ') \
                    or line.startswith('４ ') or line.startswith('５ ') or line.startswith('６ ') or line.startswith('７ ') \
                    or line.startswith('一 ') or line.startswith('二 ') or line.startswith('三 ') or line.startswith('四 ') \
                    or line.startswith('五 ') or line.startswith('六 ') or line.startswith('七 '):
                if num != len(sectiontitlelist) - 1 and exactline(sectiontitlelist[num + 1]) == (exactline(line) + 1):
                    line = exacttext_without_line(line) + sectiontitlelist[num + 1]
                tmp.append(line)
        if tmp == []:
            errorcount += 1
            paper.sectiontitle = None
        else:
            paper.sectiontitle = '\n'.join(tmp)
    print(journal + ': %d errors' % errorcount)
    session.commit()


def clean_sectiontitle1():
    session = DBSession()  # 创建Session:
    errorcount = 0
    papers = session.query(Paper).filter(Paper.abstract != None).filter(Paper.journal == '情报科学') \
        .filter(Paper.sectiontitle != None)
    for paper in papers:
        startnum = -1
        sectiontitlelist = paper.sectiontitle.splitlines()
        for num, line in enumerate(sectiontitlelist):
            if line.startswith('1') or line.startswith('１'):
                startnum = num
                sectiontitlelist[num] = sectiontitlelist[num].replace('１', '1')
                break
        if startnum == -1:
            errorcount += 1
        else:
            paper.sectiontitle = '\n'.join(sectiontitlelist[startnum:])
    print(errorcount)
    session.commit()


def clean_sectiontitle5():  # 无用
    session = DBSession()  # 创建Session:
    errorcount = 0
    papers = session.query(Paper).filter(Paper.abstract != None).filter(Paper.journal == '图书情报工作') \
        .filter(Paper.sectiontitle != None)
    for paper in papers:
        tmp = []
        for line in paper.sectiontitle.splitlines():
            a = find_line(line)
            if len(a) == 2:
                line = line.replace(a[0], '')
            tmp.append(line)
        paper.sectiontitle = '\n'.join(tmp)
    session.commit()


def clean_abstract_keyword():
    session = DBSession()  # 创建Session:
    papers = session.query(Paper).filter(Paper.abstract != None).filter(Paper.year == '2016').all()
    for paper in papers:
        paper.abstract = paper.abstract.lstrip('摘捅要 ［］[]〔〕【】:：()').replace('\n', '')
        paper.keyword = paper.keyword.lstrip('关键词健抽「 ［］[]〔〕【】I:：()').replace('\n', '')
    session.commit()


def exact_abstract_keyword(journal):
    session = DBSession()  # 创建Session:
    papers = session.query(Paper).filter(Paper.journal == journal).filter(Paper.abstract == None). \
        filter(Paper.year == '2016').all()
    errorcount = 0
    for paper in papers:
        abstract_linenum = None
        keyword_linenum = None
        lines = paper.frontcontent.splitlines()  # 等价于split('\n')
        for num, line in enumerate(lines):
            line = exacttext(line)
            if line.startswith('摘 要') or line.startswith('摘要') or line.startswith('〔摘') \
                    or line.startswith('［摘') or line.startswith('［ 摘') or line.startswith('[ 摘') \
                    or line.startswith('[摘') or line.startswith('【摘') or line.startswith('捅要') \
                    or line.startswith('(摘要'):
                abstract_linenum = num
            if line.startswith('关键词') or line.startswith('关键字') or line.startswith('关健词') \
                    or line.startswith('关健字') or line.startswith('〔关') or line.startswith('［关') \
                    or line.startswith('［ 关') or line.startswith('[ 关') or line.startswith('[关') \
                    or line.startswith('【关') or line.startswith('I关') or line.startswith('关键性') \
                    or line.startswith('「关') or line.startswith('关抽词') or line.startswith('关键 词') \
                    or line.startswith('关 键词') or line.startswith('(关键词'):
                keyword_linenum = num
                if abstract_linenum != None: break  # 保证找到的是在摘由后面出现关键字才停止
        if abstract_linenum and keyword_linenum:
            paper.abstract = ''.join(map(exacttext, lines[abstract_linenum:keyword_linenum]))
            paper.keyword = exacttext(lines[keyword_linenum])
        else:
            print(abstract_linenum)
            print(keyword_linenum)
            errorcount += 1
            print(paper.journal + ' | ' + paper.phase + ': ' + paper.filename)
    session.commit()
    print(journal + ': %d errors!' % errorcount)


def exact_title_by_filename(journal):
    session = DBSession()  # 创建Session:
    papers = session.query(Paper).filter(Paper.journal == journal).filter(Paper.title == None).all()
    count = 0
    for paper in papers:
        title = filename_to_title(paper.filename)
        title = paper.filename.replace('%3a', '：')  # 特例情报学报
        title = title.replace('省略', '')  # 最后10条没精力去人工替换了Orz
        if '省略' in title:
            count += 1
            continue
        paper.title = title
    session.commit()
    print(count)


def exact_title_qbzlgz():  # 提取情报资料工作标题
    session = DBSession()  # 创建Session:
    papers = session.query(Paper).filter(Paper.journal == '情报资料工作').filter(Paper.title != None).all()
    count = 0
    for paper in papers:
        title = filename_to_title(paper.filename)
        if '省略' in title:
            count += 1
            continue
        paper.title = title
    session.commit()
    print(count)


def filename_to_title(str):
    lastdash = str.rfind('_')
    result = str[:lastdash].replace('_', '')
    return result


def exact_title_qbzz():  # 情报杂志提取标题
    session = DBSession()  # 创建Session:
    papers = session.query(Paper).filter(Paper.journal == '情报杂志').filter(Paper.title == None).all()
    count = 0
    for paper in papers:
        list = paper.frontcontent.split('\n')
        for row, rowcontent in enumerate(list):
            if rowcontent.startswith('23.6678###'):
                title = clean(exacttext(rowcontent))
                nextline = clean(exacttext(list[row + 1]))
                if issubtitle(nextline):
                    title += nextline
                paper.title = title
                count += 1
                break
    session.commit()
    print(count)


def clean(str):
    result = str.rstrip(' *＊#!"').replace(' 、', '、')
    return result.lstrip()


def issubtitle(str):
    if hasalpha(str):
        return True
    elif ',' in str or '，' in str or ' ' in str or len(str) < 4 or str.endswith('1') or str.endswith('2'):
        return False
    else:
        return True


def hasalpha(str):
    my_re = re.compile(r'[A-Za-z]')
    return bool(re.search(my_re, str))


def exact_title_qbllysj():  # 抽情报理论与实践title
    session = DBSession()  # 创建Session:
    papers = session.query(Paper).filter(Paper.journal == '情报理论与实践').all()
    for paper in papers:
        list = paper.frontcontent.split('\n')
        for row, rowcontent in enumerate(list):
            if rowcontent.startswith('20.9273###'):
                title = exacttext(rowcontent).replace('*', '').replace('＊', '').rstrip()
                nextline = exacttext(list[row + 1])
                if '摘 要' not in nextline and '＊' not in nextline:
                    title += nextline
                paper.title = title
                break
    session.commit()


def exact_title_qbkx():  # 抽情报科学title
    session = DBSession()  # 创建Session:
    papers = session.query(Paper).filter(Paper.journal == '情报科学').all()
    for paper in papers:
        title = ''
        list = paper.frontcontent.split('\n')
        for row, rowcontent in enumerate(list):
            title = ''
            if rowcontent.startswith('18.0482###') and not rowcontent.startswith('18.0482###·'):
                title = exacttext(rowcontent)
                nextline = exacttext(list[row + 1])
                if ',' not in nextline and '，' not in nextline and ' 1' not in nextline and len(nextline) > 4:
                    title += nextline
                paper.title = title
                break
    session.commit()


def count():
    session = DBSession()  # 创建Session:
    papers = session.query(Paper).filter(Paper.year == '2016').all()
    print(len(papers))
    count = 0
    for paper in papers:
        if paper.frontcontent.find('27.1557###') != -1:
            count += 1
    print('count', count)
    session.close()


def genfrontcontent():  # 提取前50行当作frontcontent
    session = DBSession()  # 创建Session:

    # papers = session.query(Paper).filter(Paper.journal == '情报学报').all()
    papers = session.query(Paper).all()
    print(len(papers))
    for paper in papers:
        frontcontent = paper.content.split('\n')[:50]
        frontcontent = '\n'.join(frontcontent)
        paper.frontcontent = frontcontent
    session.commit()


def exacttext(string):  # string为已经去除了换行符
    regex = r'[\d\.]+?.*?###(.*)$'
    # pattern = re.compile(regex)
    try:
        result = re.search(regex, string).group(1)
    except:
        print('错误匹配', string)
        return string
    return result


def exactline(string):  # 返回的行数为int类型
    regex = r'#(\d+)?#$'
    try:
        result = re.search(regex, string).group(1)
    except:
        print('错误匹配', string)
        return string
    return int(result)


def exacttext_without_line(string):  # 返回的行数为int类型
    regex = r'^(.*)#\d+?#$'
    try:
        result = re.search(regex, string).group(1)
    except:
        print('错误匹配', string)
        return string
    return result


def find_line(string):
    regex = r'(#\d+?#).*(#\d+?#)$'
    try:
        result = re.search(regex, string).groups()
    except:
        return (None,)
    return result


def test_cleancontent():
    session = DBSession()  # 创建Session:
    cleancontents = session.query(Paper.cleancontent).filter(Paper.year == '2016').filter(Paper.journal == '现代情报').all()
    for i in range(5):
        for l in cleancontents[i].splitlines():
            print(l)
        print()
    session.close()


def main():
    '''
     # 创建session对象:
    session = DBSession()
    # 创建新User对象:
    new_paper = Paper(filename='你好', journal='你好', year='1221', phase='2', content='中文中文')
    new_paper1 = Paper(filename='你好', journal='你好', year='1221', phase='2', content='中文中文')
    # 添加到session:
    session.add(new_paper)
    session.add(new_paper1)
    # 提交即保存到数据库:
    session.commit()
    # 关闭session:
    session.close()
    '''

    # count()
    # genfrontcontent()

    # session = DBSession()  # 创建Session:
    # papers = session.query(Paper).filter(Paper.journal == '情报理论与实践' and Paper.year == '2016').all()
    # for paper in papers:
    #     paper.content = paper.content.replace('; sans-serif','')
    # session.commit()
    # clean_abstract_keyword()

    # exact_sectiontitle_qbzlgz()
    # journals = [
    #     '情报科学', '情报理论与实践', '情报学报', '情报杂志', '情报资料工作',
    #     '图书情报工作', '图书情报知识', '图书与情报', '现代情报', '现代图书情报技术'
    # ]
    # for journal in journals:
    #     exact_title_by_filename(journal)
    #     exact_abstract_keyword(journal)
    #     if journal == '现代情报':
    #     exact_abstract_keyword(journal)
    #     exact_zhengwen(journal)
        # exact_sectiontitle(journal)
        # clean_sectiontitle4(journal)

        # gen_cleancontent(journal)


        #    if journals[i + 1] != '情报学报':
        #        continue
        #    exact_zhengwen(journals[i + 1])
        # exact_zhengwen('情报学报')
        # exact_title_by_filename('情报学报')
        # exact_abstract_keyword('情报资料工作')


        # clean_abstract_keyword()

        # exact_sectiontitle('情报学报')
        # clean_sectiontitle2('情报科学')
        # exact_zhengwen('情报资料工作')
        # exact_sectiontitle_qbzlgz()
        # gen_cleancontent()
        # fenci()
        # clean_keyword('情报学报')
        # combine_stopword()
        # filter_by_wordlen()
        # collect_keyword()
        # addfreq()
        # fenci_abstract()
        # filter_by_wordlen()
        # fenci_by_column()
        # split_content()
        # wordcount()
        # gen_wordlist_file()
        # testtest()
        # cal_DF2()
        # sort_DF()
        # select_by_DF()
        # total_word_clean()
        # cal_TFIDF2()
        # cal_TF()
        # sort_TForDF_bylen()
        # cal_TFxIDF()
        # filter_word_remove_English()
        # gen_corpus_file()
        # cal_TF_word2vec()
        # normalization_by_file()
        # wordcount()
        # gen_paperid_list()
        # cal_word_sum()
        # select_by_DF()
        # cal_TFIDF_word2vec()
        # normalization_l1_by_file()
        # normalization_l2_by_file()

if __name__ == '__main__': main()
