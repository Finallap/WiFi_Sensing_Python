import requests
import re
from bs4 import BeautifulSoup
import jieba
from wordcloud import WordCloud

#三个新闻url
# url = 'https://news.sina.cn/2019-12-23/detail-iihnzahi9459965.d.html'
# url = 'https://news.sina.cn/2019-12-23/detail-iihnzahi9449867.d.html'
url = 'https://news.sina.cn/gn/2019-12-26/detail-iihnzhfz8356984.d.html'

#使用requests模块获取网页，并用BeautifulSoup解析出p标签内容
r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')
all_p = soup.find_all("p", attrs={"class": "art_p"})

#提取p标签内容，去除标签括号
s = ''
for item in all_p:
    l_text = re.findall('<p class="art_p">(.*)</p>', str(item))
    if l_text != []:
        s = s + l_text[0]
    l_text = re.findall('<p class="art_p" cms-style="font-L">(.*)</p>', str(item))
    if l_text != []:
        s = s + l_text[0]

#使用jieba模块分词
words = jieba.lcut(s)
dict_words = {}
for word in words:
    if len(word)>1:
        dict_words[word] = dict_words.get(word, 0) + 1

#使用WordCloud模块生成词云
wd_obj = WordCloud(font_path='msyh.ttc')
wd_obj.generate_from_frequencies(dict_words)
wd_obj.to_image()
wd_obj.to_file('url3.png')