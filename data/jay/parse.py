#-*- coding:utf-8 -*-
import collections
import codecs
import re

'''
with codecs.open('lyrics.txt','r',encoding='utf-8') as f:
    content = f.read()
c = collections.Counter(content)
words = [x[0] for x in c.most_common()]
print words[-1]
'''
new = []
key = [u'周杰伦',u'方文山',u'作词',u'作曲',u'编曲',u'王羲之']
with codecs.open('origin.txt','r',encoding='utf-8') as f:
    content = f.readlines()
    for c in content:
	if True:
            reg = re.compile(ur"[\s+]")
            c = reg.sub(' ',unicode(c))
            reg = re.compile(ur"[^\u4e00-\u9fa5\s]")
            c = reg.sub('',unicode(c))
	    c = c.strip()
	    #if len(c)!=0 and u'周杰伦' not in c and u'方文山' not in c:
	    if len(c)!=0 and unicode(c)!=u'歌词' and all(i not in c for i in key):
	        c = c.replace('  ',' ')
	        c = c+' '
	        new.append(c)

with codecs.open('new.txt','w','utf-8') as f:
    for x in new:
        f.write(x)
