import LocusPlotting
from matplotlib import pyplot as plt
num = [1]
denum = [1,125,5100,65000,0];
#num=[1];
#denum=[1,4,8,0]
LocusPlotting.root_locus(num,denum,Plot=True);
plt.show()