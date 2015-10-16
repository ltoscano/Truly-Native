import re
from urllib.parse import urlparse
import itertools
import nltk
import numpy as np
from nltk.corpus import stopwords # Import the stop word list
from bs4 import Comment

# Utilities

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
word_detector = nltk.data.load('tokenizers/punkt/english.pickle')
stop_words    = set(stopwords.words("english"))

def purge_html(soup):
    """ Remove unwanted html content (javascript, comments, css)
    """
    for tag in soup(['script', 'style', 'link', 'meta', 'iframe']):
        tag.extract()
    for comment in soup.findAll(text=lambda text:isinstance(text, Comment)):
        comment.extract()
    return soup
                
def html2txt(soup):
    """ Improves bs4's get_text() using a purged html
    """
    # cleanup 
    for tag in soup(['i', 'b', 'span', 'a']):
        tag.unwrap()
    textnodes = soup.findAll(text=True)
    for i in range(0, len(textnodes)-1):
        if textnodes[i].next_sibling == textnodes[i+1]:
            textnodes[i+1].string = textnodes[i].string + " " + textnodes[i+1].string
            textnodes[i].extract()
    return soup.get_text('\n', strip=True)
        
# Feature extractors

def is_internal_link(url):
    relative = re.compile(r"^/|^\.|^\#")
    if relative.match(url) != None:
        return True
    return False

def count_links(soup):
    nbIntLinks = 0
    nbExtLinks = 0
    for tag in soup('a'):
        if 'href' in tag.attrs:
            if is_internal_link(str(tag['href'])):
                nbIntLinks = nbIntLinks + 1
            else:
                nbExtLinks = nbExtLinks + 1
    return (nbIntLinks, nbExtLinks)

def avg_sentence_len(text):
    global sent_detector
    strings = sent_detector.tokenize(text.strip())
    lengths = list(map(lambda s : len(s.strip()) -1, strings))
    lengths = list(filter(lambda l : l > 0, lengths))
    lengths = np.fromiter(lengths, dtype=np.int)
    return np.mean(lengths)

def get_clean_words(text):
    """ return most significant words
    """
    global stopwords
    letters = re.sub("[^a-zA-Z]", " ", text) 
    text    = text.lower().split()
    # 5. Remove stop words
    meaningful_words = [w for w in words if w not in stops]  
    return nltk.FreqDist(text1)