# some changes have made after Gensim-3.x to 4 
# https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4

from gensim.models import Word2Vec
from tqdm import tqdm

def getVocabCount(model_path):
    model = Word2Vec.load(model_path)
    vocab_count = len(model.wv)
    return vocab_count

def genKeyVec(model_path, model_output_path):
    '''
        Output the key vector of trained Word2Vec model 
        first line : # of vocabulary # of vector_size
        second line to end : key vector pair
        e.g.
            655247 300
            字詞 -0.042409 0.139688 -0.091641 0.126242 -0.108757 0.052762 -0.082010 ...   
            字詞 -0.068804 0.068888 -0.057192 0.047150 -0.092274 -0.069476 -0.101947 ...   
    '''
    vocab_count = getVocabCount(model_path)
    print('Output key vector to "{}"'.format(model_output_path))
    model = Word2Vec.load(model_path)

    pbar = tqdm(total=vocab_count)
    with open(model_output_path, 'w', encoding='utf-8') as f:
        f.write('%d %d\n' % (vocab_count, 300))
        for word in list(model.wv.index_to_key):
            f.write('%s %s\n' % (word, ' '.join([str(round(v, 6)) for v in model.wv[word]])))
            pbar.update(1)
            break
    pbar.close()    

if __name__ == '__main__':
    model_path = '../model/word2vec-healthdoc-wiki-300.model'
    vector_output_path = '../model/healthdoc-wiki-300.vector'
    vocab_count = getVocabCount(model_path)
    print('Total number of vocabulary : {}'.format(vocab_count))
    genKeyVec(model_path, vector_output_path)
