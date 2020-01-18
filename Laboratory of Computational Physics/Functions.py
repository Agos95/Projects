import numpy  as np
import pandas as pd
from math import fabs
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy import stats
from scipy.optimize import curve_fit
from os.path import exists

# Cell dimensions
XCELL = 42.
ZCELL = 13.

# X coordinates translation
global_x_shifts = [994.2, 947.4,-267.4,-261.5,]

# Z coordinates translations
local_z_shifts = [z*ZCELL for z in range(0,4)]
global_z_shifts = [823.5, 0, 823.5, 0]

def Read_Data(event_str):
    '''Read input event as string and create the Dataframe

    Parameters
    ----------
    event_str : str
        Single line of the data file

    Returns
    -------
    dataframe : pandas.dataframe
        Pandas Dataframe with all the information about the Event;
        each row is a single hit recorded in the event
    event_number : int
        number of the Event in the run
    hits_number : int
        number of hits in the Event
    '''

    event        = event_str.split()
    event_number = int(event[0])
    hits_number  = int(event[1])
    if hits_number == 0:
        hit       = [np.nan]
        chamber   = [np.nan]
        layer     = [np.nan]
        xl_local  = [np.nan]
        xr_local  = [np.nan]
        z_local   = [np.nan]
        time      = [np.nan]
        xl_global = [np.nan]
        xr_global = [np.nan]
        z_global  = [np.nan]
    else:
        hit       = np.arange(hits_number)
        chamber   = np.fromiter((event[2+5*i] for i in range(hits_number)), int)
        layer     = np.fromiter((event[3+5*i] for i in range(hits_number)), int)
        xl_local  = np.fromiter((event[4+5*i] for i in range(hits_number)), float)
        xr_local  = np.fromiter((event[5+5*i] for i in range(hits_number)), float)
        z_local   = np.fromiter((local_z_shifts[i-1]+ZCELL/2 for i in layer), float)
        time      = np.fromiter((event[6+5*i] for i in range(hits_number)), float)
        xl_global = np.fromiter((global_x_shifts[i] for i in chamber), float) - xl_local
        xr_global = np.fromiter((global_x_shifts[i] for i in chamber), float) - xr_local
        z_global  = np.fromiter((global_z_shifts[i] for i in chamber), float) + z_local
    dataframe = pd.DataFrame({
        'EvNumber' : event_number,
        'Hit'      : hit,
        'Chamber'  : chamber,
        'Layer'    : layer,
        'XL_local' : xl_local,
        'XR_local' : xr_local,
        'Z_local'  : z_local,
        'Time'     : time,
        'XL_global': xl_global,
        'XR_global': xr_global,
        'Z_global' : z_global,
        })
    return dataframe, event_number, hits_number

def Plot_Background():
    '''Makes the plot for the background of the event display

    Returns
    -------
    axes : list(pyplot.axes)
        background images of the detector and the chambers
    '''

    # create Pandas DataFrame for the chambers positions
    chamber_position = pd.DataFrame({
    'chamber' : [i for i in range(4)],
    'x_vertices' : [(global_x_shifts[i], global_x_shifts[i] - 720, global_x_shifts[i] - 720, global_x_shifts[i])
                    for i in range(4)],
    'y_vertices' : [(global_z_shifts[i], global_z_shifts[i], global_z_shifts[i] + 52, global_z_shifts[i] + 52)
                    for i in range(4)],
    })
    x_lim = [[-1000, 1000], # global detector
                [    0, 1000], # chamber 0
                [    0, 1000], # chamber 1
                [-1000,    0], # chamber 2
                [-1000,    0]] # chamber 3
    y_lim = [[-100, 1000],  # global detector
                [800 ,  900],  # chamber 0
                [ -25,   75],  # chamber 1
                [ 800,  900],  # chamber 2
                [ -25,   75]]  # chamber 3
    title = ["DETECTOR", "Chamber 0", "Chamber 1", "Chamber 2", "Chamber 3"]
    # create pyplot 'Axes' objects
    gridsize = (5,2)
    ax_global = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
    ax_0 = plt.subplot2grid(gridsize, (2, 1), colspan=1, rowspan=1) # top-right
    ax_1 = plt.subplot2grid(gridsize, (3, 1), colspan=1, rowspan=1) # bottom-right
    ax_2 = plt.subplot2grid(gridsize, (2, 0), colspan=1, rowspan=1) # top-left
    ax_3 = plt.subplot2grid(gridsize, (3, 0), colspan=1, rowspan=1) # bottom-left

    axes = [ax_global, ax_0, ax_1, ax_2, ax_3]

    if exists("./wire_pos_glob.txt"):
        wires = np.loadtxt("./wire_pos_glob.txt")
    else: wires = None

    for index, ax in enumerate(axes):
        ax.set_xlim(x_lim[index])
        ax.set_ylim(y_lim[index])
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        if index == 0: ax.set_title(title[index])
        else: ax.set_title(title[index], pad=-20)
        # plot the 4 chambers in each 'Axes'
        for j in range(4):
            chamber = chamber_position[chamber_position["chamber"] == j]
            ax.fill(chamber["x_vertices"].values[0], chamber["y_vertices"].values[0], color='gray', fill=False)
        if wires is not None:
            ax.plot(wires[:,0], wires[:,1], marker=".", markersize=.5, linestyle="", color="gray")
    return axes

def Plot_Events(dataframe, evNumber):
    '''Plot the positions of the hits

    Parameters
    ----------
    dataframe : pandas.dataframe
        Pandas Dataframe with all the information about the Event;
        each row is a single hit recorded in the event
    event_number : int
        number of the Event in the run

    Returns
    -------
    event_display : list(pyplot.axes)
        images of the hits of the events
    '''

    # get the EvNumber as argument, because, if the dataframe is empty,
    # I can't get it from data
    plots = Plot_Background()
    plots[0].set_title("Event: {:d}".format(evNumber), {'size':'18'})
    if dataframe.empty == False:
        xL = dataframe["XL_global"]
        xR = dataframe["XR_global"]
        z  = dataframe["Z_global"]
        for image in plots:
            image.plot(xL, z, "bo", markersize=3)
            image.plot(xR, z, "ro", markersize=3)

    return plots

def Plot_Fit(ev, plots, calibration=False):
    # obtaine object from event dictionary
    dataframe = ev["Dataframe"]
    evNumber  = ev["Number"]
    chambers  = ev["Chambers"]
    layers    = ev["Layers"]

    # if the list of plots is empty, create it
    if not plots: plots = Plot_Events(dataframe, evNumber)

    z_glob=np.array([x for x in range(0,1001)])

    # calibration: plot local fit and global fit (using local selected points)
    if calibration:
        lf_results = Local_Fit(dataframe, chambers, layers, exclusion_layer=True)
        for idx, ch in enumerate(chambers):
            #slope
            slope_loc = -lf_results[idx]['slope'] #in global coordinates the local slope takes a minus sign
            #intercept in global coordinates
            intercept_loc = global_x_shifts[ch]-lf_results[idx]['intercept']-slope_loc*global_z_shifts[ch]
            for image in plots:
                image.plot(slope_loc*z_glob+intercept_loc, z_glob, linestyle='--', color="C9")
        gf_results = [Global_Fit_Calibration(dataframe, chambers, lf_results)]
        for gf in gf_results:
            slope=gf['slope']
            intercept=gf['intercept']
            for image in plots:
                image.plot(slope*z_glob+intercept, z_glob, color='navy')
    else:
        df_left, df_right, _, _ = Points_Filter(dataframe)
        gf_results_left  = Global_Fit_Physics(df_left , layers[0])
        gf_results_right = Global_Fit_Physics(df_right, layers[1])
        for gf in [gf_results_left, gf_results_right]:
            slope = gf['slope']
            intercept = gf['intercept']
            for image in plots:
                image.plot(slope*z_glob+intercept, z_glob, color='navy')

    return plots


def Make_Plot(ev, calibration=False):
    '''Plots of the background and the hits

    Parameters
    ----------
    ev : dict
        dictionary with information about the event; it contains: Number, Dataframe, Hits, Accepted, Chambers, Layers
    calibration : bool
        wether the event is taken during calibration or physics run
    '''

    #gridsize = (5, 2)
    plt.figure(figsize = (12, 24))
    plots = Plot_Events(ev["Dataframe"], ev["Number"])
    if ev["Accepted"]: Plot_Fit(ev, plots, calibration=calibration)
    plt.show()
    return

def Select_Events_Calibration(dataframe, hits_number):
    '''Select good Events (calibration)
    Parameters
    ----------
    dataframe : pandas.dataframe
        Pandas Dataframe with all the information about the Event;
        each row is a single hit recorded in the event
    hits_number : int
        number of hits in the Event
    Returns
    -------
    select : bool
        True if the event pass the selection, False otherwise
    chambers : list(int)
        list with the number of the chambers where we find the hits
    n_layer : list(int)
        list with the number of different layers hit per chamber
    '''

    # hits only in the right side
    # if we have less than 6 hits or more than 20
    # mark the event as 'bad'
    if (hits_number < 6 or hits_number > 12):
            select=False
            chambers=[]
            n_layer=[]

    else:
        #hits only in the right side
        if((dataframe['Chamber']<=1).all()):
            chambers=[0,1]
            #compute number of different layers in each chamber
            n_layer_ch0 = dataframe[dataframe['Chamber']==0]['Layer'].nunique()
            n_layer_ch1 = dataframe[dataframe['Chamber']==1]['Layer'].nunique()

            n_layer=[n_layer_ch0, n_layer_ch1]

            #require at least 3 different layers for each chamber
            if(n_layer_ch0>=3 and n_layer_ch1>=3):
                select=True
                #return select, chambers, n_layer
            else:
                select=False
                #return select, chambers, n_layer

        #hits only in the left side
        elif((dataframe['Chamber']>=2).all()):
            chambers=[2,3]
            #compute number of different layers in each chamber
            n_layer_ch2 = dataframe[dataframe['Chamber']==2]['Layer'].nunique()
            n_layer_ch3 = dataframe[dataframe['Chamber']==3]['Layer'].nunique()

            n_layer=[n_layer_ch2, n_layer_ch3]

            #require at least 3 different layers for each chamber
            if(n_layer_ch2>=3 and n_layer_ch3>=3):
                select=True
            else:
                select=False

        #hits in both left and right side
        else:
            select=False
            chambers=[]
            n_layer=[]

    return select, chambers, n_layer

def Points_Filter(dataframe):

    # left detector part (chamber 2,3)
    df_left  = dataframe[dataframe["Chamber"]>=2]
    df_left_filtered = df_left[df_left['XL_local']>=0]
    df_left_filtered = df_left_filtered[df_left_filtered['XL_local']<=200]
    number_hits_left  = len(df_left_filtered)

    # right detector part (chamber 0,1)
    df_right = dataframe[dataframe["Chamber"]<=1]
    df_right_filtered = df_right[df_right['XL_local']>=540]
    df_right_filtered = df_right_filtered[df_right_filtered['XL_local']<=720]
    number_hits_right = len(df_right_filtered)

    return df_left_filtered, df_right_filtered, number_hits_left,number_hits_right

def Local_Fit(dataframe, list_chambers, list_layers, exclusion_layer=False):
    ''' Local fit for each pair of chambers (left or right side); best combination of points is the one with the lowest chi^2

    Parameters
    ----------
    dataframe : pandas.dataframe
        Pandas Dataframe with the information of the left (right) part of the detector (2 chambers)
    list_chambers : list(int)
        list with the number of the 2 chambers involved
    list_layers : list(list(int))
        list with 2 lists with the number of different layers per chamber
    exclusion_layer : bool
        wether to exclude or not the 4th layer at random when fitting (if present); if False, the excluded layer is set to 0, otherwise it is in [1,4]

    Returns
    -------
    results : list(dict)
        list of 2 dictionaries, one for each chamber; each dictionary has the following information: slope, intercept, optimal combination of points for the fit, excluded layer

    '''
    #list to store results for each chamber
    results = []
    #loop over the (two) chambers
    for i in range(0,len(list_chambers)):
       #if we have 4 different layers we randomly select a layer to be excluded
       #we will use the point from the excluded layer to check the goodness of the global fit
        if(list_layers[i]==4 and exclusion_layer==True):
            rand_layer=np.random.randint(1,5)
        else:
            rand_layer=0 #layers are 1,2,3,4: excluding layer 0 is equivalent to keeping them all

        #create dataframe_cl filtered by chamber and excluded layer
        dataframe_c = dataframe[dataframe['Chamber']==list_chambers[i]] #dataframe filtered by chamber
        dataframe_cl = dataframe_c[dataframe_c['Layer']!=rand_layer]    #filtered by chamber and excluded layer

        # Z local coordinates corresponding to the 4 different layers
        Z=[6.5, 19.5, 32.5, 45.5]

        #create a list l containing lists of points (z,x), one for each selected layer
        l=[]

        #loop over selected layers and fill l
        for layer_index in dataframe_cl['Layer'].unique():
            XR=np.array(dataframe_cl[dataframe_cl['Layer']==layer_index]['XR_local'])
            XL=np.array(dataframe_cl[dataframe_cl['Layer']==layer_index]['XL_local'])

            z=Z[(layer_index-1)] #layer_index is in range [1,4], list index must be in range [0,3]
            l_temp=[]

            for x in XR:
                l_temp.append((z,x))
            for x in XL:
                l_temp.append((z,x))
            l.append(l_temp)

        #create numpy array with all possible combinations of 3 (4) points p1,p2,p3(,p4)
        if(list_layers[i]==3 or exclusion_layer==True):
            combinations=np.array([(p1,p2,p3) for p1 in l[0] for p2 in l[1] for p3 in l[2]])
        elif(list_layers[i]==4 and exclusion_layer==False):
            combinations=np.array([(p1,p2,p3,p4) for p1 in l[0] for p2 in l[1] for p3 in l[2] for p4 in l[3]])
        else:
            print("ERROR, Unexpected number of layers")
            break

        #interpolate each combination and select the combination with least chi squared
        min_chisq=100000 #to store minimum chisq
        if(list_layers[i]==3 or exclusion_layer==True):
            optimal_comb=np.zeros((3,2)) #to store best combination of points
        if(list_layers[i]==4 and exclusion_layer==False):
            optimal_comb=np.zeros((4,2)) #to store best combination of points
        slope_opt=0 #to store slope obtained with the best combination
        intercept_opt=0 #to store intercept obtained with the best combination
        for points in combinations:
            #linear regression
            slope, intercept, _, _, _ = stats.linregress(points[:,0],points[:,1])
            #compute expected x using the interpolating function
            expect_x=intercept+slope*(points[:,0])
            #compute chi squared
            chisq, _ = stats.chisquare(points[:,1],expect_x)
            #eventually update min_chisq and optimal_comb
            if(chisq<min_chisq):
                min_chisq = chisq
                optimal_comb = points
                slope_opt = slope
                intercept_opt = intercept
            else:
                continue

        #add to results: results is a list of 2 dictionaries, one for each chamber
        results.append({"slope":slope_opt,
                        "intercept":intercept_opt,
                        "optimal_comb": optimal_comb,
                        "excl_layer": rand_layer})

    return results

def Global_Fit_Calibration(dataframe, list_chambers, lfit_results):
    ''' Global fit for left (right) side using the points selected by Local_Fit

    Parameters
    ----------
    dataframe : pandas.dataframe
        Pandas Dataframe with the information of the left (right) part of the detector (2 chambers)
    list_chambers : list(int)
        list with the number of the 2 chambers involved
    lfit_results : list(dict)
        list of 2 dictionaries, one for each chamber; each dictionary has the following information: slope, intercept, optimal combination of points for the fit, excluded layer. This is the results from Local_Fit

    Returns
    -------
    g_results : dict
        dictionary with the following information:
        {
            slope : float - slope of the fitting line
            intercept : float - intercept of the fitting line
            residuals : np.array([z1,x1], [z2,x2]) - residuals from points in the excluded layer, if present;
                        otherwise it is empty
        }
    '''

    #TRANSFORM LOCAL COORDINATES IN GLOBAL COORDINATES

    #First chamber:
    global_z_ch1 = global_z_shifts[list_chambers[0]]+lfit_results[0]["optimal_comb"][:,0]
    global_x_ch1 = global_x_shifts[list_chambers[0]]-lfit_results[0]["optimal_comb"][:,1]
    global_ch1   = np.column_stack((global_z_ch1, global_x_ch1))

    #Second chamber:
    global_z_ch2 = global_z_shifts[list_chambers[1]]+lfit_results[1]["optimal_comb"][:,0]
    global_x_ch2 = global_x_shifts[list_chambers[1]]-lfit_results[1]["optimal_comb"][:,1]
    global_ch2   = np.column_stack((global_z_ch2, global_x_ch2))

    points = np.concatenate((global_ch1, global_ch2))

    #LINEAR REGRESSION
    slope, intercept, _, _, _ = stats.linregress(points[:,0],points[:,1])

    #compute expected x using the interpolating function
    expect_x = intercept+slope*(points[:,0])

    #COMPUTE RESIDUALS USING TEST LAYER (layer excluded in local fit function)
    # Z local coordinates corresponding to the 4 different layers
    Z_local = [6.5,19.5, 32.5, 45.5]
    #list to store residuals
    res=[]
    #compute residuals for each chamber
    for c in range(0,len(list_chambers)):
        dataframe_c = dataframe[dataframe['Chamber']==list_chambers[c]] #dataframe filtered by chamber
        res_temp=[]
        excl_layer=lfit_results[c]["excl_layer"]
        #test layer Z global coordinate
        Z_test_layer=global_z_shifts[c]+Z_local[(excl_layer-1)]
        #if there were only 3 layers, excl_layer was set to 0:
        if(excl_layer!=0):
            expect_x=intercept+slope*(Z_test_layer)
            XR=np.array(dataframe_c[dataframe_c['Layer']==excl_layer]['XR_global'])
            XL=np.array(dataframe_c[dataframe_c['Layer']==excl_layer]['XL_global'])
            for i in range(0,XR.size):
                res_temp.append(XR[i]-expect_x)
            for i in range(0,XL.size):
                res_temp.append(XL[i]-expect_x)

            res_temp.sort(key=fabs) #we want the smallest residual in absolute value
            res.append(res_temp[0])
        else:
            res=[]
    #convert list res in numpy array
    res=np.array(res)

    g_results = {"slope": slope, "intercept": intercept, "residuals": res }

    return g_results

#def Global_Fit_Physics(dataframe, list_chambers, list_layers):
def Global_Fit_Physics(dataframe, list_layers):
    ''' Global fit for left (right) side using all the points recorded; best combination of points is the one with the lowest chi^2

    Parameters
    ----------
    dataframe : pandas.dataframe
        Pandas Dataframe with the information of the left (right) part of the detector (2 chambers)
    list_chambers : list(int)
        list with the number of the 2 chambers involved
    list_layers : list(list(int))
        list with 2 lists with the number of different layers per chamber

    Returns
    -------
    g_results : dict
        dictionary with the following information:
        {
            slope : float - slope of the fitting line
            intercept : float - intercept of the fitting line
            optimal_comb : list([z1,x1], [z2,x2], ...) - optimal combination of points chosen for the fit
        }
    '''

    #list to store results for each chamber
    g_results=[]
    #create a list l containing lists of points (z,x), one for each selected layer
    l=[]
    # list of chambers
    list_chambers = dataframe["Chamber"].unique()
    #loop over the (two) chambers
    for i in range(0,len(list_chambers)):
        #create dataframe_cl filtered by chamber and excluded layer
        dataframe_cl = dataframe[dataframe['Chamber']==list_chambers[i]] #dataframe filtered by chamber

        # Z local coordinates corresponding to the 4 different layers
        Z=[6.5, 19.5, 32.5, 45.5]
        global_z_shifts = [823.5, 0, 823.5, 0]

        #loop over selected layers and fill l
        for layer_index in dataframe_cl['Layer'].unique():
            XR=np.array(dataframe_cl[dataframe_cl['Layer']==layer_index]['XR_global'])
            XL=np.array(dataframe_cl[dataframe_cl['Layer']==layer_index]['XL_global'])

            z= global_z_shifts[list_chambers[i]]+Z[(layer_index-1)] #layer_index is in range [1,4], list index must be in range [0,3]
            l_temp=[]

            for x in XR:
                l_temp.append((z,x))
            for x in XL:
                l_temp.append((z,x))
            l.append(l_temp)

    # print(len(l))
    #create numpy array with all possible combinations of 3 (4) points p1,p2,p3(,p4)
    #print(list_layers)
    total_layers=list_layers[0]+list_layers[1]
    if(total_layers==6):
        combinations=np.array([(p1,p2,p3,p4,p5,p6) for p1 in l[0] for p2 in l[1] for p3 in l[2] for p4 in l[3] for p5 in l[4] for p6 in l[5]])
    elif(total_layers==7):
        combinations=np.array([(p1,p2,p3,p4,p5,p6,p7) for p1 in l[0] for p2 in l[1] for p3 in l[2] for p4 in l[3] for p5 in l[4] for p6 in l[5] for p7 in l[6]])
    elif(total_layers==8):
        combinations=np.array([(p1,p2,p3,p4,p5,p6,p7,p8) for p1 in l[0] for p2 in l[1] for p3 in l[2] for p4 in l[3] for p5 in l[4] for p6 in l[5] for p7 in l[6] for p8 in l[7]])
    else:
        print("ERROR, Unexpected number of layers")
        g_results = {"slope" : np.nan,
                     "intercept" : np.nan,
                     "residuals" : []}
        return g_results

    #interpolate each combination and select the combination with least chi squared
    min_chisq=100000 #to store minimum chisq
    """if(total_layers==6):
        optimal_comb=np.zeros((6,2)) #to store best combination of points with hit in 6 layer
    elif(total_layers==7):
        optimal_comb=np.zeros((7,2)) #to store best combination of points with hit in 7 layer
    elif(total_layers==8):
        optimal_comb=np.zeros((8,2)) #to store best combination of points with hit in 8 layer"""

    slope_opt=0 #to store slope obtained with the best combination
    intercept_opt=0 #to store intercept obtained with the best combination
    for points in combinations:
        #linear regression
        slope, intercept, _, _, _ = stats.linregress(points[:,0],points[:,1])
        #compute expected x using the interpolating function
        expect_x = intercept+slope*(points[:,0])
        #compute chi squared
        chisq, _ = stats.chisquare(points[:,1],expect_x)
        #eventually update min_chisq and optimal_comb
        if(fabs(chisq) < min_chisq):
            min_chisq=fabs(chisq)
            residuals = np.subtract(points[:,1], expect_x).tolist()
            slope_opt=slope
            intercept_opt=intercept
        else:
            continue

    #add to results
    g_results = {"slope" : slope_opt,
                 "intercept" : intercept_opt,
                 "residuals" : residuals}

    return g_results

def Gaussian_Fit_Hist(ax, data, nbins=None, hist_range=None, **kwargs):

    def Gaussian(x,A, mu, sigma):
        return A * np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2))

    hist, bins, _ = ax.hist(data, bins=nbins, range=hist_range, **kwargs)
    centers = (bins[:-1] + bins[1:]) / 2
    #area = sum(np.diff(bins)*hist)
    popt3, pcov3 = curve_fit(Gaussian, xdata=centers, ydata=hist)
    A, mu, sigma = popt3
    dA, dmu, dsigma = [np.sqrt(pcov3[j,j]) for j in range(popt3.size)]
    x = np.linspace(bins.min(),bins.max(), 100)

    ax.plot(x, Gaussian(x, A, mu, sigma), color='C1', linestyle="-.", linewidth=2, label=r'$f(x) = A\,e^{-(x-\mu)^2/2\sigma^2}$')
    ax.errorbar(centers, hist, ecolor="red", elinewidth=1, fmt="ro", markersize=.75, yerr=np.sqrt(hist))
    textstr = '\n'.join(('$A$ = {0:0.2f}$\pm${1:0.2f}'.format(A, dA),
                         '$\mu$ = {0:0.4f}$\pm${1:0.4f}'.format(mu, dmu),
                         '$\sigma$ = {0:0.4f}$\pm${1:0.4f}'.format(sigma, dsigma)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=15,
              verticalalignment='top', bbox=props)
    #return A, mu, sigma, dA, dmu, dsigma, centers
    return