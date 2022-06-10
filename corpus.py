import os
import string
import torch
import pickle
import math
import time
import requests
import read_healthdoc
from tqdm import tqdm
from opencc import OpenCC
from zhon.hanzi import punctuation
from ckip_transformers.nlp import CkipWordSegmenter
from gensim.corpora.wikicorpus import WikiCorpus

def download_file(url, path):
    # download the file form url
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0
    print('Downloading %s to %s', url, path)
    with open(path, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size / block_size), unit='KB', unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)
    print('Done')

def download_wiki_dump(path):
    # Download the latest chinese wiki dump file : https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2
    url = 'https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2'
    if not os.path.exists(path):
        print("Start downloading latest zh wiki dump...")
        download_file(url, path)
    else:
        print('"{file}" exists, skip download'.format(file=path))

def createPKL_healthdoc(healthdoc_path, pklfile_path):
    # Word segmentation results of healthdoc in pkl format
    # The pkl file store :
    #   first line : the size of the file (e.g. 2724, total documents in healthdoc)
    #   second line to end : list of word segmentation result of each document
    #   e.g.
    #       2724 
    #       ['核災','食品', '／', '買到', '「', '核災區', '」', '食品', '？', ...])

    #pklfile_path = '../dataset/healthdoc.pkl'  # pkl file store at the root dir
    # check if there exist healthdoc.pkl, continue it if true 
    file_exists = os.path.exists(pklfile_path)
    if file_exists:
        print('"{file}" exists, skip build'.format(file=pklfile_path))
        return

    print('Build "{file}"'.format(file=pklfile_path))
    #
    healthdoc = read_healthdoc.loading(healthdoc_path)
    s2t = OpenCC('s2t')
    if (torch.cuda.is_available()):
        ws_driver = CkipWordSegmenter(level=3, device=0) # word segementation tool, device use GPU
        print("Using GPU to word segmentation")
    else:
        ws_driver = CkipWordSegmenter(level=3) # without GPU
    #
    total_size = len(healthdoc)
    with open(pklfile_path, 'ab') as pkf: # put the length of the file at the beinging
        pickle.dump(total_size, pkf)
    pbar = tqdm(total=total_size) # total=total_size, unit=' doc', unit_scale=False

    # punctuation of en and zh
    punctuation_en = string.punctuation
    punctuation_zh = punctuation
    digits = string.digits
    
    #
    for k, v in healthdoc.items():    
        doc = ("".join(item for item in v.split('\n') if item)) # remove '\n' in the document
        doc = s2t.convert(doc) # convert simplified Chinese into traditional Chinese
        doc_ws = ws_driver([doc], show_progress=False) # word segmentation

        # remove punctuation
        temp = []
        healthdoc_rm_pun = []
        for word in doc_ws[0]:
            if word in punctuation_en or word in punctuation_zh or word in digits:
                continue
            else:
                temp.append(word)
        healthdoc_rm_pun.append(temp)
        
        # save list to pkl file
        with open(pklfile_path, 'ab') as pkf:
            #pickle.dump(doc_ws[0], pkf)
            pickle.dump(healthdoc_rm_pun[0], pkf)
        pbar.update(1)    
        temp = []
        healthdoc_rm_pun = []        
    torch.cuda.empty_cache()

def createPKL_wiki(wiki_dump_path, pklfile_path):
    # Word segmentation results of wiki in pkl format
    # The pkl file store :
    #   first line : the size of the file (e.g. 427601, total documents in zh-wiki)
    #   second line to end : list of word segmentation result of each document
    #   e.g.
    #       427601 
    #       ['歐幾裏得', '西元', '前', '三世紀', '的', '古希臘', '數學家', ...]

    #pklfile_path = '../dataset/zhwiki.pkl'  # pkl file store at the root dir
    # check if there exist zhwiki.pkl, continue if true 
    file_exists = os.path.exists(pklfile_path)
    if file_exists:
        print('"{file}" exists, skip build'.format(file=pklfile_path))
        return

    print('Build "{file}"'.format(file=pklfile_path))
    #
    print("Parsing Wiki corpus...", end='')
    t = time.time()
    wiki = WikiCorpus(wiki_dump_path)
    print("cost [%d:%d:%d]"%((time.time()-t)/3600, ((time.time()-t)%3600)/60, (time.time()-t)%60))

    s2t = OpenCC('s2t')
    if (torch.cuda.is_available()):
        ws_driver = CkipWordSegmenter(level=3, device=0) # word segementation tool, device use GPU
        print("Using GPU to word segmentation")
    else:
        ws_driver = CkipWordSegmenter(level=3) # without GPU

    #
    total_size = wiki.length
    pbar = tqdm(total=total_size) 
    with open(pklfile_path, 'ab') as pkf:
        pickle.dump(total_size, pkf)  

    for sentence in wiki.get_texts(): 
        doc = ''
        for word in sentence: # combine all words in a document to a string
            doc = doc + word
        doc = s2t.convert(doc) # convert simplified Chinese into traditional Chinese
        doc_ws = ws_driver([doc], show_progress=False) # word segmentation 

        with open(pklfile_path, 'ab') as pkf:
            pickle.dump(doc_ws[0], pkf)
        pbar.update(1)          
                                          
    print("Done")
    torch.cuda.empty_cache()


class CorpusSentencesPKL:
  # reference: https://github.com/LasseRegin/gensim-word2vec-model/blob/master/train.py
  # Data iterator for training Word2Vec
  # each iter will return a list of word segments in a document
  def __init__(self, wiki_pkl_path, healdoc_pkl_path):
    print('Parsing wiki corpus')
    self.wiki_pkl_path = wiki_pkl_path
    print('Parsing healthdoc corpus')
    self.healdoc_pkl_path = healdoc_pkl_path

  def __iter__(self):    
    #
    print('Processing HealthDoc')
    healthdoc_pkl = open(self.healdoc_pkl_path, "rb")
    total_size = pickle.load(healthdoc_pkl) # get size of healthdoc_pkl
    pbar = tqdm(total=total_size)
    for doc in range(total_size):
        doc_ws=pickle.load(healthdoc_pkl, encoding='utf-8')
        pbar.update(1)
        yield(doc_ws)
    pbar.close()

    #
    print('Processing Wik-zh')
    wiki_pkl = open(self.wiki_pkl_path, "rb")
    total_size = pickle.load(wiki_pkl) # get size of wiki_pkl
    pbar = tqdm(total=total_size)
    for doc in range(total_size):
        doc_ws=pickle.load(wiki_pkl, encoding='utf-8')
        pbar.update(1)
        yield(doc_ws)
    pbar.close()