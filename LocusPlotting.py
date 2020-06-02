import matplotlib
import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import pylab  # plotting routines
from bisect import *
def _TransferFunction(Num,Den):
    """Converting Num and Den to Polynomials"""
    num=poly1d(Num);
    den=poly1d(Den);
    return num,den;

def root_locus(Num,Den,Plot=True):
    plotstr ='b' if int(matplotlib.__version__[0]) == 1 else 'C0'
    grid =True;
    kvect=None;
    xlim=None;
    ylim=None;

    # Convert numerator and denominator to polynomials if they aren't
    (nump, denp) = _TransferFunction(Num,Den);
    start_mat = _FindRoots(nump, denp, [1])
    kvect, mymat, xlim, ylim = _gains(nump, denp, xlim, ylim)
    Kbreak,rBreak=_break_points(nump,denp);
    # Show the Plot
    if Plot:
        figure_number = pylab.get_fignums()
        figure_title = [
            pylab.figure(numb).canvas.get_window_title()
            for numb in figure_number]
        new_figure_name = "Root Locus"
        rloc_num = 1
        while new_figure_name in figure_title:
            new_figure_name = "Root Locus " + str(rloc_num)
            rloc_num += 1
        f = pylab.figure(new_figure_name)
        ax = pylab.axes()
        ax.plot(rBreak,np.zeros(len(rBreak)),'x',color='red')
        # plot open loop poles appears in X in red
        poles = array(denp.r)
        #Making the pole point with random colors to be diffrent
        for x,y in zip(real(poles),imag(poles)):
            rgb=(np.random.rand(),np.random.rand(),np.random.rand());
            ax.scatter(x,y,c=[rgb]);
        _DepratureAngels(f,poles);


        # plot open loop zeros if existed appears in O in black #just for trial reasons
        zeros = array(nump.r)
        if zeros.size > 0:
            ax.plot(real(zeros), imag(zeros), 'x' ,color='black')
        # Now plot the loci
        for index, col in enumerate(mymat.T):
            ax.plot(real(col), imag(col), plotstr, label='root locus')
        # Set up plot axes and labels
        if xlim:
          ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        ax.set_xlabel('Re')
        ax.set_ylabel('Im')
        if grid:
            _grid_func(f,num=nump,den=denp);
        elif grid:
            _grid_func(num=nump,den=denp)

        ax.axhline(0., linestyle=':', color='gray')
        ax.axvline(0., linestyle=':', color='gray')



def _gains(num, den, xlim, ylim):
    k_break, real_break = _break_points(num, den)
    print(real_break,"real break",k_break,"k_break")
    #upperbound to the gains
    kmax = _k_max(num, den, real_break, k_break)
    #KVect contains range of gains
    kvect = np.hstack((np.linspace(0, kmax, 50), np.real(k_break)))
    kvect.sort()
    #My Matrix Contains the Roots over the gain range in kvect
    mymat = _FindRoots(num, den, kvect)
    mymat = _SortRoots(mymat)
    open_loop_poles = den.roots
    open_loop_zeros = num.roots
    #Filling The zeros to make it the same size as poles
    if open_loop_zeros.size != 0 and \
            open_loop_zeros.size < open_loop_poles.size:
        open_loop_zeros_xl = np.append(
            open_loop_zeros,
            np.ones(open_loop_poles.size - open_loop_zeros.size)
            * open_loop_zeros[-1])
        mymat_xl = np.append(mymat, open_loop_zeros_xl)
    else:
        mymat_xl = mymat
    singular_points = np.concatenate((num.roots, den.roots), axis=0)
    important_points = np.concatenate((singular_points, real_break), axis=0)
    mymat_xl = np.append(mymat_xl, important_points)

    false_gain = float(den.coeffs[0]) / float(num.coeffs[0])
    # The graph window limits
    if xlim is None and false_gain > 0:
        x_tolerance = 0.05 * (np.max(np.real(mymat_xl))
                              - np.min(np.real(mymat_xl)))
        xlim = _ax_lim(mymat_xl)

    elif xlim is None and false_gain < 0:
        axmin = np.min(np.real(important_points)) \
                - (np.max(np.real(important_points))
                   - np.min(np.real(important_points)))
        axmin = np.min(np.array([axmin, np.min(np.real(mymat_xl))]))
        axmax = np.max(np.real(important_points)) \
                + np.max(np.real(important_points)) \
                - np.min(np.real(important_points))
        axmax = np.max(np.array([axmax, np.max(np.real(mymat_xl))]))
        xlim = [axmin, axmax]
        x_tolerance = 0.05 * (axmax - axmin)
    else:
        x_tolerance = 0.05 * (xlim[1] - xlim[0])

    if ylim is None:
        y_tolerance = 0.05 * (np.max(np.imag(mymat_xl))
                              - np.min(np.imag(mymat_xl)))
        ylim = _ax_lim(mymat_xl * 1j)
    else:
        y_tolerance = 0.05 * (ylim[1] - ylim[0])

    # Figure out which points are spaced too far apart
    if x_tolerance == 0:
        # Root locus is on imaginary axis , use just y distance
        tolerance = y_tolerance
    elif y_tolerance == 0:
        # Root locus is on imaginary axis , use just x distance
        tolerance = x_tolerance
    else:
        tolerance = np.min([x_tolerance, y_tolerance])

    indexes_too_far = _filt(mymat, tolerance)
    # Add more points into the root locus for points that are too far apart
    while len(indexes_too_far) > 0 and kvect.size < 5000:
        for counter, index in enumerate(indexes_too_far):
            index = index + counter * 3
            #creating new subGains
            new_gains = np.linspace(kvect[index], kvect[index + 1], 5)
            #adding new subPoints
            new_points =_FindRoots(num, den, new_gains[1:4])
            kvect = np.insert(kvect, index + 1, new_gains[1:4])
            mymat = np.insert(mymat, index + 1, new_points, axis=0)
        mymat = _SortRoots(mymat)
        indexes_too_far = _filt(mymat, tolerance)

    #To see when we have some high gains to make the graph continue after xlim and ylim
    new_gains = kvect[-1] * np.hstack((np.logspace(0, 3, 4)))
    new_points = _FindRoots(num, den, new_gains[1:4])
    #print(new_gains,"new Points with high gains");
    kvect = np.append(kvect, new_gains[1:4])
    mymat = np.concatenate((mymat, new_points), axis=0)
    mymat = _SortRoots(mymat)

    return kvect, mymat, xlim, ylim

def _filt(mymat, tolerance):
    #getting points in the gap between points
    distance_points = np.abs(np.diff(mymat, axis=0))
    indexes_too_far = list(np.unique(np.where(distance_points > tolerance)[0]))
    indexes_too_far = list(np.unique(indexes_too_far))
    indexes_too_far.sort()
    return indexes_too_far


def _break_points(num, den):
    """Extract break points over real axis and gains given these locations"""
    # type: (np.poly1d, np.poly1d) = (np.array, np.array)
    #taking dervatives
    dnum = num.deriv()
    dden = den.deriv()
    polynom = den * dnum - num * dden
    real_break_pts = polynom.r
    real_break_pts = real_break_pts[num(real_break_pts) != 0]
    k_break = -den(real_break_pts) / num(real_break_pts)
    idx = k_break >= 0  # only positives gains
    k_break = k_break[idx]
    real_break_pts = real_break_pts[idx]
    if len(k_break) == 0:
        k_break = [0]
        real_break_pts = den.roots
    return k_break, real_break_pts


def _ax_lim(mymat):
    """Utility to get the axis limits"""
    axmin = np.min(np.real(mymat))
    axmax = np.max(np.real(mymat))
    if axmax != axmin:
        deltax = (axmax - axmin) * 0.02
    else:
        deltax = np.max([1., axmax / 2])
    axlim = [axmin - deltax, axmax + deltax]
    return axlim


def _k_max(num, den, real_break_points, k_break_points):
    """"Calculate the maximum gain for the root locus with asymptotes(estimation)"""
    asymp_number = den.order - num.order
    singular_points = np.concatenate((num.roots, den.roots), axis=0)
    important_points = np.concatenate(
        (singular_points, real_break_points), axis=0)
    false_gain = den.coeffs[0] / num.coeffs[0]
    if asymp_number > 0:
        asymp_center = (np.sum(den.roots) - np.sum(num.roots)) / asymp_number
        #estimation
        distance_max = 4 * np.max(np.abs(important_points - asymp_center))

        asymp_angles = (2 * np.arange(0, asymp_number) + 1) \
                       * np.pi / asymp_number

        if false_gain > 0:
            # farthest points over asymptotes from the face that  e^(theta*j)=cos(theta)+sin(theta)j
            farthest_points = asymp_center \
                              + distance_max * np.exp(asymp_angles * 1j)
        else:
            asymp_angles = asymp_angles + np.pi
            # farthest points over asymptotes
            farthest_points = asymp_center \
                              + distance_max * np.exp(asymp_angles * 1j)

        #getting KMax estimation
        kmax_asymp = np.real(np.abs(den(farthest_points)
                                    / num(farthest_points)))
    else:
        kmax_asymp = np.abs([np.abs(den.coeffs[0])
                             / np.abs(num.coeffs[0])*3])

    kmax = np.max(np.concatenate((np.real(kmax_asymp),
                                  np.real(k_break_points)), axis=0))
    if np.abs(false_gain) > kmax:
        kmax = np.abs(false_gain)
    return kmax

def _FindRoots(nump, denp, kvect):
    """Find the roots for the root locus."""
    roots = []
    for k in kvect:
        curpoly = denp + k * nump
        curroots = curpoly.r
        if len(curroots) < denp.order:
            # if I have fewer poles than open loop, it is because i have
            # one at infinity
            curroots = np.insert(curroots, len(curroots), np.inf)

        curroots.sort()
        roots.append(curroots)

    mymat = row_stack(roots)
    return mymat


def _SortRoots(mymat):
    """Sort the roots """

    sorted = zeros_like(mymat)
    for n, row in enumerate(mymat):
        if n == 0:
            sorted[n, :] = row
        else:
            # sort the current row by finding the element with the
            # smallest absolute distance to each root in the
            # previous row
            available = list(range(len(prevrow)))
            for elem in row:
                evect = elem - prevrow[available]
                ind1 = abs(evect).argmin()
                ind = available.pop(ind1)
                sorted[n, ind] = elem
        prevrow = sorted[n, :]
    return sorted
def _removeLine(label, ax):
    """Remove a line from the ax when a label is specified"""
    for line in reversed(ax.lines):
        if line.get_label() == label:
            line.remove()
            del line


def _grid_func(fig=None, zeta=None, wn=None,num=None,den=None):
    "Drawing zetaLines and Asymptotes"
    if fig is None:
        fig = pylab.gcf()

    ax = fig.gca();
    xlocator = ax.get_xaxis().get_major_locator()
    print( xlocator()[1],"dddd")
    ylim = ax.get_ylim()
    ytext_pos_lim = ylim[1] - (ylim[1] - ylim[0]) * 0.03
    xlim = ax.get_xlim()
    xtext_pos_lim = xlim[0] + (xlim[1] - xlim[0]) * 0.0

    if zeta is None:
        zeta = _zetas(xlim, ylim)
    if(den!=None and num.order !=None):
        asymp_number = den.order - num.order
        false_gain = den.coeffs[0] / num.coeffs[0]
        if asymp_number > 0:
            asymp_center = (np.sum(den.roots) - np.sum(num.roots)) / asymp_number
            asymp_angles = (2 * np.arange(0, asymp_number) + 1) \
                           * np.pi / asymp_number
            if false_gain>=0:
                asymp_angles = asymp_angles + np.pi
            yn=np.tan(asymp_angles);
            for y in yn:
             ax.plot([real(asymp_center), xlocator()[0]], [imag(asymp_center),
                    y * (xlocator()[0]-real(asymp_center))], color='blue',
                    linestyle=':', linewidth=1)
             ax.plot([real(asymp_center), xlim[1]], [imag(asymp_center),
                                                             -y*(xlim[1]-real(asymp_center))], color='blue',
                    linestyle=':', linewidth=1)

    angules = []
    for z in zeta:
        if (z >= 1e-4) and (z <= 1):
            angules.append(np.pi / 2 + np.arcsin(z))
        else:
            zeta.remove(z)
    y_over_x = np.tan(angules)

    # zeta-constant lines

    index = 0
    for yp in y_over_x:
        ax.plot([0, xlocator()[0]], [0, yp * xlocator()[0]], color='gray',
                linestyle=':', linewidth=0.5)
        ax.plot([0, xlocator()[0]], [0, -yp * xlocator()[0]], color='gray',
                linestyle=':', linewidth=0.5)
        an = "%.2f" % zeta[index]
        if yp < 0:
            xtext_pos = 1 / yp * ylim[1]
            ytext_pos = yp * xtext_pos_lim
            if np.abs(xtext_pos) > np.abs(xtext_pos_lim):
                xtext_pos = xtext_pos_lim
            else:
                ytext_pos = ytext_pos_lim
            ax.annotate(an, textcoords='data', xy=[xtext_pos, ytext_pos],
                        fontsize=8)
        index += 1

    ax.plot([0, 0], [ylim[0], ylim[1]],
            color='gray', linestyle='dashed', linewidth=0.5)

    angules = np.linspace(-90, 90, 20) * np.pi / 180
    if wn is None:
        wn = _wn(xlocator(), ylim)

    for om in wn:
        if om < 0:
            yp = np.sin(angules) * np.abs(om)
            xp = -np.cos(angules) * np.abs(om)
            ax.plot(xp, yp, color='gray',
                    linestyle='dashed', linewidth=0.5)
            an = "%.2f" % -om
            ax.annotate(an, textcoords='data', xy=[om, 0], fontsize=8)


def _zetas(xlim, ylim):
    """Return default list of dumps coefficients to draw lines in the grid"""
    sep1 = -xlim[0] / 4
    ang1 = [np.arctan((sep1 * i) / ylim[1]) for i in np.arange(1, 4, 1)]
    sep2 = ylim[1] / 3
    ang2 = [np.arctan(-xlim[0] / (ylim[1] - sep2 * i)) for i in np.arange(1, 3, 1)]

    angules = np.concatenate((ang1, ang2))
    angules = np.insert(angules, len(angules), np.pi / 2)
    zeta = np.sin(angules)
    return zeta.tolist()


def _wn(xloc, ylim):
    """Return default wn for root locus plot"""

    wn = xloc
    sep = xloc[1] - xloc[0]
    while np.abs(wn[0]) < ylim[1]:
        wn = np.insert(wn, 0, wn[0] - sep)

    while len(wn) > 7:
        wn = wn[0:-1:2]

    return wn

def _low(poles):
    min_val = 10000000
    j=-1;
    for i in poles:
        if min_val>np.real(i) and np.real(i)<0:
            min_val=np.real(i);
            j=i;
    return min_val,j;
def _high(poles):
    min_val = 10000000
    j=-1;
    for i in poles:
        if min_val>np.real(i) and np.real(i)>0:
            min_val=np.real(i);
            j=i;
    return min_val,j;




def _DepratureAngels(fig,poles):
    ax=fig.gca()
    complex_poles = [];
    for i in poles:
        if (np.imag(i) != 0):
            complex_poles.append(i);
    complex_poles = array(complex_poles);
    final_angels = [];
    for i in complex_poles:
        angels = [];
        for j in poles:
            if (np.real(i) == np.real(j) and np.imag(i) == np.imag(j)):
                continue;
            plt.plot([np.real(i), np.real(j)], [np.imag(i),
                                                np.imag(j)], color='red',
                     linestyle=':', linewidth=1)
            offset=np.pi;
            if (np.real(i) - np.real(j) == 0):
                x=np.pi/2;
                if(np.imag(i)<0):
                    x+=offset;

            else:
                slope = (np.imag(j) - np.imag(i)) / (np.real(j) - np.real(i));
                x=np.arctan(slope)
                if(np.imag(i)<0):
                    x+=offset;
            print(np.rad2deg(x),"angles")
            angels.append(x);
        final_angels.append(np.pi - np.sum(angels));
    final_angels=np.rad2deg(final_angels)
    for i in range(len(final_angels)):
        while(final_angels[i]<0):
            final_angels[i]+=360;
    counter=0;
    for xy in zip(np.real(complex_poles),np.imag(complex_poles)):
        ax.annotate('( %0.2f )' % final_angels[counter], xy=xy, textcoords='data',color='b'
                    ,size=7.5)
        counter+=1;

