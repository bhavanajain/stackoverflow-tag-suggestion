
# coding: utf-8

# In[1]:


"""
A generator that yields a chunk of dataset file using the specified delimiter
"""

def myreadlines(f, delimiter):
  buf = ""
  while True:
    while delimiter in buf:
      pos = buf.index(delimiter)
      yield buf[:pos]
      buf = buf[pos + len(delimiter):]
    chunk = f.read(4096)
    if not chunk:
      yield buf
      break
    buf += chunk


# In[2]:


question_delimiter = "\n|||||\n"
category_delimiter = "\n;;;;;\n"


# In[3]:


"""
Create a dictionary of tags with key = tag name and value = number of occurrences
"""

curr_tags = ""
tags_occur = {}

with open('./dataset.txt', 'r') as f:
  for post in myreadlines(f, question_delimiter):
    categories = post.split(category_delimiter)
    if len(categories) > 2:
        curr_tags = categories[1].lower()
        for tag in curr_tags.split(" "):
            if tag not in tags_occur:
                tags_occur[tag] = 0 
            tags_occur[tag] += 1

tags_count = sorted(set(tags_occur.values()), reverse=True)
req_idx = 100
min_count = tags_count[req_idx]
top_tags = [k for k, v in tags_occur.items() if v > min_count]
tags_map = {k : i for (i, k) in enumerate(top_tags)}
print("Number of tags %d" % len(tags_map.keys()))
print(len(tags_occur))

# In[4]:


from bs4 import BeautifulSoup
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = stopwords.words('english')
question_body = ""
f_doc = open('processed_docs.txt', 'w')
f_labels = open('labels.txt', 'w')
X = []
Y = []

with open('./dataset.txt') as f:
  for post in myreadlines(f, question_delimiter):
    categories = post.split(category_delimiter)
    if len(categories) > 2:
        given_tags = set(categories[1].lower().split(" ")) # Ground truth
        pruned_tags = [tag for tag in given_tags if tag in top_tags]
        if len(pruned_tags) == 0:
            continue
        y = [0] * len(top_tags)
        for tag in pruned_tags:
            y[tags_map[tag]] = 1
            
        question_body = categories[2]
        soup = BeautifulSoup(question_body, 'html.parser')

        # Remove all tags with a class or id containing the word snippet 
        # Later use these snippets to predict the programming language ^_^
        for snippet_tag in soup.find_all(attrs={'class': re.compile('snippet')}):
            snippet_tag.decompose()
        for snippet_tag in soup.find_all(attrs={'id': re.compile('snippet')}):
            snippet_tag.decompose()

        # Remove all the <pre> ... </pre> tags
        for extra in soup('pre'):
            extra.extract()

        tokens = word_tokenize(soup.get_text().lower())
        filtered_tokens = []
        for (i, token) in enumerate(tokens):
            # Remove ['.', '?', ',', '!', ':'] at the end of a token
            if re.match(re.compile('(\d+\.?)+'), token):
                continue
            if re.match(re.compile('^\W+$'), token):
                continue
            while token[-1] in ['.', '?', ',', '!', ':', ';', "'", '"']:
                token = token[:-1]
            if token not in stop_words and len(token) > 1:
                filtered_tokens.append(token)
        
        processed_body = ' '.join(filtered_tokens)
        f_doc.write(processed_body + '\n')
        f_doc.flush()
        f_labels.write(' '.join(list(map(str, y))) + '\n')
        f_labels.flush()
        X.append(processed_body)
        Y.append(y)
