import WordLemmaHelper
from Vocabs import _EOS, _UNK
from Vocabs import _EOS_ID, _UNK_ID
from collections import namedtuple


DatasetSplit = namedtuple('Dataset', ['train', 'dev', 'test'])
def prepare_data(data_dir, max_vocab_size, tokenizer=None):
  """Get WMT data into data_dir, create vocabularies and tokenize data.

  Args:
    data_dir: directory in which the data sets will be stored.
    max_vocab_size: size limit of the word  / lemma vocab
    fr_vocabulary_size: size of the French vocabulary to create and use.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.

  Returns:
    A tuple of 6 elements:
      (1) path to the token-ids for English training data-set,
      (2) path to the token-ids for French training data-set,
      (3) path to the token-ids for English development data-set,
      (4) path to the token-ids for French development data-set,
      (5) path to the English vocabulary file,
      (6) path to the French vocabulary file.
  """

  data_dir = '../../corpus/es-en'
  # the lemma2word files contain both the "source" (lemmas, POS, features) and the target (words) data.
  train_path =  "/small.train.tok.morph.es.parse"
  dev_path = "/small.dev.tok.morph.es.parse"
  test_path = "/small.test.tok.morph.es.parse"

  WordLemmaHelper.create_vocabs(data_dir, train_path, max_vocab_size, data_dir, train_path)
  vocabs = WordLemmaHelper.read_vocabs(data_dir, train_path)

  train = get_parallel_lemma2word(data_dir, train_path, vocabs)
  dev   = get_parallel_lemma2word(data_dir, dev_path, vocabs)
  test  = get_parallel_lemma2word(data_dir, test_path, vocabs)

  return DatasetSplit(train, dev, test), vocabs

class Struct():
    pass

def get_parallel_lemma2word(data_dir, train_path, vocabs):
    filename = data_dir + train_path
    S_tokens = []
    T_tokens = []
    source_tokens = []
    target_tokens = []

    S_ids = []
    T_ids = []
    source_ids = []
    target_ids = []
    for i, (word, lemmas, features) in enumerate(WordLemmaHelper.yield_records(filename)):
        if word is None and lemmas is None and features is None:
            if len(source_ids) > 0 and len(target_ids) > 0:
                S_ids.append(source_ids)
                T_ids.append(target_ids)
                S_tokens.append(source_tokens)
                T_tokens.append(target_tokens)
            source_ids = []
            target_ids = []
            source_tokens = []
            target_tokens = []
        else:
            # source tokens to ids
            lemma_ids = [vocabs.lemmas.get(lemma, _UNK_ID) for lemma in lemmas]
            feature_ids = [[vocabs.features.get(f, _UNK_ID) for f in F] for F in features]  # features = [F1, F2, F2] = [[f1, f2], [..] ....]
            source_ids.append((lemma_ids, feature_ids))
            source_tokens.append((lemmas, features))
            # target words to ids
            word_id = vocabs.words.get(word, _UNK_ID)
            target_ids.append(word_id)
            target_tokens.append(word)

    parallel_data = Struct()
    parallel_data.source_tokens = S_tokens
    parallel_data.source_ids = S_ids
    parallel_data.target_tokens = T_tokens
    parallel_data.target_ids = T_ids

    return parallel_data
