#!/usr/bin/env python3
"""reducer_1.py"""

from operator import itemgetter
import sys
import csv

current_word = None
current_count = 0
word = None

file1 = open('final_output.txt', 'a') 

L = ["\n\n The no. of tasks deployed to every node using map reduce:\n\n",
"Node no. \t No. of tasks\n\n"]

file1.writelines(L)

# input comes from STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()

    # parse the input we got from mapper.py
    word, count = line.split('\t', 1)

    # convert count (currently a string) to int
    try:
        count = int(count)
    except ValueError:
        # count was not a number, so silently
        # ignore/discard this line
        continue

    # this IF-switch only works because Hadoop sorts map output
    # by key (here: word) before it is passed to the reducer
    if current_word == word:
        current_count += count
    else:
        if current_word:
            # write result to STDOUT

            
            s1 =  str(current_word) + "\t\t\t" + str(current_count) + "\n"
            
            file1.writelines(s1)
            
        current_count = count
        current_word = word






# do not forget to output the last word if needed!
if current_word == word:

	
	s1 =  str(current_word) + "\t\t\t" + str(current_count) + "\n"

	file1.writelines(s1) 
	
	
	
file1.close()


