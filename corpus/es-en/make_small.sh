# convert newline separated records to ~ separated records, each in its own line
# 100000 lines corresponds to roughly 5000 records
head -n 100000 train.es-en.tok.morph.es.parse | perl -ne 'chomp; if (/^\s*$/) { print "\n"} else {print "$_~"}' > small.tok.morph.es.parse
# make split
head -n 3000 small.tok.morph.es.parse | tr '~' '\n' > small.train.tok.morph.es.parse 
head -n 4000 small.tok.morph.es.parse | tail -n 1000 | tr '~' '\n' > small.dev.tok.morph.es.parse 
head -n 5000 small.tok.morph.es.parse | tail -n 1000 | tr '~' '\n' > small.test.tok.morph.es.parse 
