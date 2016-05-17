#!/usr/bin/env python

# Found a python module to fix the output from GRAMPAL

import sys
import codecs
from ftfy import fix_text
import fileinput

#thx stackoverflow
UTF8Reader = codecs.getreader('utf-8')
sys.stdin = UTF8Reader(sys.stdin)
#stdout
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

for line in sys.stdin:
    print(fix_text(line.rstrip()))

