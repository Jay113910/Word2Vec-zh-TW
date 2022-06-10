import os
import time
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

import corpus

#%%
class LossLogger(CallbackAny2Vec):
    # Output loss at each epoch
    # Reference:
    #   https://datascience.stackexchange.com/questions/9819/number-of-epochs-in-gensim-word2vec-implementation
    # Loss issue: loss tally maxes at 134217728.0 due to float32 limited-precision
    #   https://github.com/RaRe-Technologies/gensim/issues/2735
    def __init__(self):
      self.epoch = 1
      self.losses = []

    def on_epoch_begin(self, model):
      print('\nEpoch: {}'.format(self.epoch))

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 1:
            print('	Loss after epoch {}: {}'.format(self.epoch, loss))
        else:
            print('	Loss after epoch {}: {}'.format(self.epoch, float(loss) - float(self.loss_previous_step)))
        self.epoch += 1
        self.loss_previous_step = float(loss)
                  
#%%
# main function
if __name__ == '__main__':
    if not os.path.isdir('../dataset'):
        os.mkdir('../dataset')
    if not os.path.isdir('../model'):
        os.mkdir('../model')

    # path of dataset 
    wiki_dump_path = '../dataset/zhwiki.xml.bz2' 
    healthdoc_path = '../dataset/HealthDoc/'
    
    # path of dataset in pkl format
    wiki_pkl_path = '../dataset/zhwiki.pkl'
    healthdoc_pkl_path = '../dataset/healthdoc.pkl'

    model_output_path = '../model/word2vec-healthdoc-wiki-300.model'

    # model parameter
    vector_size = 300;
    workers = 4
    epochs = 5
    min_count = 10 # vocabulary under min_count would be ignore
    loss_logger = LossLogger()

    # download latest wiki dump
    corpus.download_wiki_dump(wiki_dump_path)

    # build pkl dataset for HealthDoc and zh-wiki dump
    corpus.createPKL_healthdoc(healthdoc_path, healthdoc_pkl_path)
    corpus.createPKL_wiki(wiki_dump_path, wiki_pkl_path)

    # parse corpus
    print("Parsing corpus")
    sentences = corpus.CorpusSentencesPKL(wiki_pkl_path, healthdoc_pkl_path)

    print('Training Word2Vec model')
    print('Parameter summary :\n\tvector_size : {}\n\tworkers : {}\n\tepochs : {}\n\tmin_count : {}'.format(vector_size, workers, epochs, min_count))
    t = time.time()
    model = Word2Vec(sentences, 
                     sg=1, 
                     vector_size=vector_size, 
                     workers=workers, 
                     epochs=epochs, 
                     min_count=min_count, 
                     callbacks=[loss_logger], 
                     compute_loss=True)

    print('Training done. cost [{:d}:{:d}:{:d}]'.format(int((time.time()-t)/3600), int(((time.time()-t)%3600)/60), int((time.time()-t)%60)))
    print('Save trained word2vec model to "{}"'.format(model_output_path))
    model.save(model_output_path)
    print('Done')