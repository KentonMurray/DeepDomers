#!/usr/bin/env python

import fileinput
import sys

def SplitLine(line) :
  #print >>sys.stderr, type(line)
  #print  >>sys.stderr, "in=%i %s" % (len(line), line) 
  i = 1
  while i < len(line):
    c = line[i-1]
    #print >>sys.stderr, type(c)
    sys.stdout.write(c.encode('utf-8'))
    if line[i] == ' ':
      sys.stdout.write(" ")
      i += 1
    else:
      sys.stdout.write("@@ ")
    i += 1

  sys.stdout.write(line[len(line)-1])
  print

for line in fileinput.input():
  #print type(line)
  SplitLine(line.decode("utf-8").strip())

