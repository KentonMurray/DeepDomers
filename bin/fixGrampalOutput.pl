#!/usr/bin/env perl

# Fix issues with GRAMPAL output
#
# 
#
#
use strict;
use utf8;

binmode(STDIN, ":utf8");
binmode(STDOUT, ":utf8");
binmode(STDERR, ":utf8");

while(my $row = <STDIN>) {

    #fix mis-encoded chars

    #Ãº -> é;
    $row =~ s/Ãº/é/g;
    #Ã³ -> é
    $row =~ s/Ã³/é/g;
    #Ã© -> é
    $row =~ s/Ã©/é/g;

    #Ã ->
    $row =~ s/Ã/É/g;

    #Â¿ -> ¿
    $row =~ s/Â¿/¿/g;
    
    #look for </a>...    
    #if($row =~ /<\/a>(.*)/) {
    #$row =~ s/$1/\n$1/;
    #}

    # </a>
    $row =~ s/<\/a>/<\/a>\n/g;

    # </c>
    $row =~ s/<\/c>/<\/c>\n/g;

    # <f id="1-1">
    $row =~ s/1-1\">/1-1\">\n/g;
    
    print $row;

}

#don't forget to run sed to clean up blank lines...
