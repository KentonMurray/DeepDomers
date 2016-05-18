__author__ = 'tomerlevinboim'

import codecs
import data_utils

class Vocabs():
    pass

def write_vocab(path, tokens):
    outf = codecs.open(path, 'w', 'utf-8')
    for w in tokens:
        outf.write(w + "\n")
    outf.close()


def write(outdir, output_basename, wordVocab, lemmaVocab, featVocab, charVocab):
    output_basename = outdir + output_basename
    write_vocab(output_basename + ".vocab.words", wordVocab)
    write_vocab(output_basename + ".vocab.lemmas", lemmaVocab)
    write_vocab(output_basename + ".vocab.features", featVocab)
    write_vocab(output_basename + ".vocab.chars", charVocab)

def read(path, basename):
    vocabs = Vocabs()
    vocabs.words, vocabs.rev_words       = data_utils.initialize_vocabulary(path + basename + '.vocab.words')
    vocabs.lemmas, vocabs.rev_lemmas     = data_utils.initialize_vocabulary(path + basename + '.vocab.lemmas')
    vocabs.features, vocabs.rev_features = data_utils.initialize_vocabulary(path + basename + '.vocab.features')
    vocabs.chars, vocabs.rev_chars       = data_utils.initialize_vocabulary(path + basename + '.vocab.chars')
    return vocabs