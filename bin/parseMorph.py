#!/usr/bin/env python

#import xml.etree.ElementTree as ET

import argparse
import codecs

def parse(s):
    word = (s.split('>')[1]).split('<')[0]
    returns = word
    
    try:
        lemma = (s.split("lem=\"")[1]).split('\"')[0]
    except:
        lemma = returns

    pos = (s.split("cat=\"")[1]).split('\"')[0]
    try:
        tags = ((s.split('lem=\"')[1]).split('\"'))[1:-2]
    except:
        tags = []
    
    returns += "\t"+lemma+"+"+pos
    for i in xrange(len(tags)/2):
        if tags[2*i] != " for=" and tags[2*i] != " id=":
            returns += "+"+tags[2*i+1]

    return returns.strip()

def main():
    parser = argparse.ArgumentParser(description='Parse XML output from GRAMPAL')
    parser.add_argument('-i', required=True, type=str, help='Input file')
    parser.add_argument('-o', required=True, type=str, help='Output file')

    args = parser.parse_args()
    with codecs.open(args.i, 'r', 'utf-8') as inf:
        lines = inf.readlines()
    outf = codecs.open(args.o, 'w', 'utf-8')
    
    prevw=""
    skip=False
    try:
        for i,_ in enumerate(lines):
            line = lines[i].strip()
            # New sentence
            if line == "<texto>":
                outf.write("\n")
            # Word in sentence
            elif line.strip()[:2] == "<w":
                outf.write(parse(line.strip())+"\n")
            elif line.strip()[:3] == "<c ":
                word = (line.split('>')[1]).split('<')[0]
                lemms = (line.split('<w '))
                #print lemms, len(lemms)
                if len(lemms)==4:
                    outf.write(word+'\t'+parse(lemms[1])+"|"+parse(lemms[2])+"|"+parse(lemms[3])+"\n")
                elif len(lemms)==3:
                    outf.write(word+'\t'+parse(lemms[1])+"|"+parse(lemms[2])+"\n")
                elif len(lemms)==2:
                    outf.write(word+'\t'+parse(lemms[1])+"\n")
                else:
                    print len(lemms)
                    print i, lines[i]

            elif line.strip()[:3] == "<a ":
                word = (line.split('>')[1]).split('<')[0]
                lemms = (line.split('<w '))
                #print lemms, len(lemms)
                if len(lemms)==4:
                    outf.write(word+'\t'+parse(lemms[1])+"|"+parse(lemms[2])+"|"+parse(lemms[3])+"\n")
                elif len(lemms)==3:
                    outf.write(word+'\t'+parse(lemms[1])+"|"+parse(lemms[2])+"\n")
                elif len(lemms)==2:
                    outf.write(word+'\t'+parse(lemms[1])+"\n")
                else:
                    print len(lemms)
                    print i, lines[i]
                
    except:
        print i, lines[i]
    outf.close()



if __name__ == "__main__":
    main()

