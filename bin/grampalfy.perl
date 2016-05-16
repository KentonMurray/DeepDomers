#!/usr/bin/env perl

# read input file, query GRAMPAL

use strict;
use utf8;
use Time::HiRes qw(usleep);
use URI::Escape;

binmode(STDIN, ":utf8");
binmode(STDOUT, ":utf8");
binmode(STDERR, ":utf8");


# URL base for GRAMPAL
my $urlbase = "http://cartago.lllf.uam.es/grampal/grampal.cgi";


while(my $row=<STDIN>) {

    chomp $row;
    my $enc = uri_escape($row);
    
    #build POST-friendly CURL call

    my $result = `curl -s -d fs=xml2 -d m=xml -d texto="$enc" $urlbase `;

    print "$result\n";
    usleep(1000);

    #progress!
    print STDERR ".";
    
}
