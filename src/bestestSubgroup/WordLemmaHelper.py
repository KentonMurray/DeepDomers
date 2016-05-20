#!/usr/bin/env python
import codecs
import argparse
import Vocabs
from Vocabs import _PAD, _GO, _EOS, _UNK


def initVocab():
    return [_PAD, _GO, _EOS, _UNK]


# Gets input line and returns word, lemmas, features
# returns <string>, list of strings (lemmas), and list of lists (features)
def parse_line(line):
    word, rest = line.split("\t")
    allfeats = rest.split("|")
    lemmas = []
    features = []
    for f in allfeats:
        temp = f.split("+")
        lemmas.append(temp[0])
        features.append(temp[1:])
    return word, lemmas, features


def add2Dict(d, w):
    if w in d:
        d[w] += 1
    else:
        d[w] = 1


def yield_records(filename):
    with codecs.open(filename, 'r', 'utf-8') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                yield None, None, None  # record separator
            else:
                word, lemmas, features = parse_line(line)
                yield word, lemmas, features


# Reads file line by line
def create_vocabs(input_dir, input_filename, maxVocabSize, out_dir, output_basename):
    if not input_dir.endswith('/'): input_dir += '/'
    if not out_dir.endswith('/'): out_dir += '/'

    wordVocab = initVocab()
    lemmaVocab = initVocab()
    featVocab = initVocab()
    charVocab = initVocab()

    # with codecs.open(input_dir + input_filename, 'r', 'utf-8') as inf:
    #     lines = inf.readlines()

    # count word, lemma, feature and character occurrences
    wordCount = {}
    lemmaCount = {}
    featCount = {}
    charCount = {}
    # for k, line in enumerate(lines):
    #     line = line.strip()
    #     if len(line) > 0:
    #         word, lemmas, features = parse_line(line)

    for word, lemmas, features in yield_records(input_dir + input_filename):
        if word is None and lemmas is None and features is None:
            continue # skip record separator

        add2Dict(wordCount, word)
        for lemma in lemmas:
            add2Dict(lemmaCount, lemma)
        for feats in features:
            for input_filename in feats:
                add2Dict(featCount, input_filename)
        for c in word:
            add2Dict(charCount, c)

    print "Created vocabularies: |word|=%d, |lemma|=%d |character|=%d, |feature|=%d" % (len(wordCount), len(lemmaCount), len(charCount), len(featCount))

    wordVocab += sorted(wordCount, key=wordCount.get, reverse=True)
    lemmaVocab += sorted(lemmaCount, key=lemmaCount.get, reverse=True)
    featVocab += sorted(featCount, key=featCount.get, reverse=True)
    charVocab += sorted(charCount, key=charCount.get, reverse=True)

    # chop the word and lemma vocabs to the maximum size
    wordVocab = wordVocab[:maxVocabSize]
    lemmaVocab = lemmaVocab[:maxVocabSize]

    Vocabs.write(out_dir, output_basename, wordVocab, lemmaVocab, featVocab, charVocab)


def read_vocabs(path, basename):
    return Vocabs.read(path, basename)

def main():
    parser = argparse.ArgumentParser(description='Read input file and create lookups')
    parser.add_argument('-i', required=True, type=str, help='Input file')
    parser.add_argument('-m', required=True, type=int, help='Max vocab size')
    parser.add_argument('-o', required=True, type=str, help='Output dir')
    # parser.add_argument('-o', required=True, type=str, help='Output file')
    #Read args
    args = parser.parse_args()
    infile = args.i
    maxVocabSize = args.m
    outdir = args.o
    #Read and print file
    create_vocabs(infile, maxVocabSize, outdir)


if __name__ == "__main__":
    main()


