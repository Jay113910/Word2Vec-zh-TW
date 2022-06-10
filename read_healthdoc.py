import csv
import pickle
import numpy as np

def loading(dataset_path):
  id = np.empty(0)
  content = {}
  with open(dataset_path+"healthdoc_label.csv", newline='', encoding='utf-8') as f:
    rows = csv.reader(f, delimiter=',')
    print("Read healthdoc")
    for row in rows:
      id=np.append(id, row[0])
  for file in id[1:]:
    with open(dataset_path+file, 'r', encoding="utf-8") as f:
        content[file]=f.read()
  return content

def loadHealthdocPKL(healdoc_pkl_path):
  healthdoc_pkl = open(healdoc_pkl_path, "rb")
  total_size = pickle.load(healthdoc_pkl) # get size of healthdoc_pkl
  doc_ws_list = []
  for doc in range(total_size):
      doc_ws=pickle.load(healthdoc_pkl, encoding='utf-8')
      doc_ws_list.append(doc_ws)
  return(doc_ws_list)
  