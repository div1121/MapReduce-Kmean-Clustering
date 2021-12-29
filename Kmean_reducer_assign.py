#/usr/bin/env python

import sys

for line in sys.stdin:
    key, value = line.split()
    print("%s\t%s" %(key,value))