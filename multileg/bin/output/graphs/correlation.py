from numpy import genfromtxt
from numpy import corrcoef
import sys
data = genfromtxt(sys.argv[1], delimiter=' ')
r = corrcoef(data[1:,1],data[1:,2])[0,1]
print("%.3f" % r).lstrip('0')