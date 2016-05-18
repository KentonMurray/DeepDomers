#!/usr/bin/env python
import codecs
import argparse

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"

_PAD_ID = 0
_GO_ID = 1
_EOS_ID = 2
_UNK_ID = 3


def initVocab():
    return [_PAD, _GO, _EOS, _UNK]


# Gets input line and returns word, lemmas, features
# returns <string>, list of strings (lemmas), and list of lists (features)
def read(l):
    word, rest = l.split("\t")
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


# Reads file line by line
def createVocabs(f, maxVocabSize, outdir):
    wordVocab = initVocab()
    lemmaVocab = initVocab()
    featVocab = initVocab()
    charVocab = initVocab()

    with codecs.open(f, 'r', 'utf-8') as inf:
        lines = inf.readlines()

    wordCount = {}
    lemmaCount = {}
    featCount = {}
    charCount = {}
    for line in lines:
        line = line.strip()
        if len(line) > 0:
            word, lemmas, features = read(line)

            add2Dict(wordCount, word)
            for lemma in lemmas:
                add2Dict(lemmaCount, lemma)
            for feats in features:
                for f in feats:
                    add2Dict(featCount, f)
            for c in word:
                add2Dict(charCount, c)

    print len(wordCount), len(lemmaCount), len(charCount), len(featCount)

    wordVocab += sorted(wordCount, key=wordCount.get, reverse=True)
    lemmaVocab += sorted(lemmaCount, key=lemmaCount.get, reverse=True)
    featVocab += sorted(featCount, key=featCount.get, reverse=True)
    charVocab += sorted(charCount, key=charCount.get, reverse=True)

    wordVocab = wordVocab[:maxVocabSize]
    lemmaVocab = lemmaVocab[:maxVocabSize]

    outf = codecs.open(outdir + "wordVocab.txt", 'w', 'utf-8')
    for w in wordVocab:
        outf.write(w + "\n")
    outf.close()
    outf = codecs.open(outdir + "lemmaVocab.txt", 'w', 'utf-8')
    for l in lemmaVocab:
        outf.write(l + "\n")
    outf.close()
    outf = codecs.open(outdir + "featVocab.txt", 'w', 'utf-8')
    for f in featVocab:
        outf.write(f + "\n")
    outf.close()
    outf = codecs.open(outdir + "charVocab.txt", 'w', 'utf-8')
    for c in charVocab:
        outf.write(c + "\n")
    outf.close()


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
    createVocabs(infile, maxVocabSize, outdir)


if __name__ == "__main__":
    main()


