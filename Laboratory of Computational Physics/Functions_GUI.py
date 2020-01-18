import PySimpleGUI as sg
font = ("bitstream charter", 14, "")
import numpy as np
import matplotlib.pyplot as plt
hist_kwargs = {"edgecolor":"black"}

from Functions import *

def Open_File_Calibration(data_file):
    with open(data_file) as f:
        tot_ev = len(list(f))

    # loop over events and perform analysis
    Ev_list = []
    selected_ev = 0
    with open(data_file) as f:
        for line in f:
            # read event
            ev, evNum, hits = Read_Data(line)
            # select event
            sel, chambers, n_layers = Select_Events_Calibration(ev, hits)

            if sel: selected_ev += 1

            # save most important information, in order to plot or reperform the analysis
            # without reading the whole file again
            Ev_list.append(
                {
                    "Number"    : evNum,
                    "Dataframe" : ev,
                    "Hits"      : hits,
                    "Accepted"  : sel,
                    "Chambers"  : chambers,
                    "Layers"    : n_layers
                }
            )
            # GUI progress bar
            sg.OneLineProgressMeter('File Loader', len(Ev_list), tot_ev, 'Reading CALIBRATION file', orientation="h", size=(30,30))

    print("{:35} = {:d}"    .format("Total number of events in the Run", tot_ev))
    print("{:35} = {:d}"    .format("Number of accepted events"        , selected_ev))
    print("{:35} = {:.4f} %".format("Fraction of accepted events"      , selected_ev/tot_ev*100))
    # create a pop up for the GUI with summary information
    sg.Popup("{:} = {:d}"    .format("Total number of events in the Run", tot_ev),
             "{:} = {:d}"    .format("Number of accepted events"        , selected_ev),
             "{:} = {:.4f} %".format("Fraction of accepted events"      , selected_ev/tot_ev*100),
            title="Summary", font=font)
    return Ev_list

def Calibration(Ev_list):
    tot_ev = len(Ev_list)
    selected_ev = 0
    # number of hits per event
    hits_number = []
    # residuals from good events fit
    Ev_residuals = []
    # x position of the best combinations of points per chamber
    X_position = {"up" : [], "down" : []}
    # local fit differences between chambers
    lf_diff = {"slope" : [], "intercept" : []}

    print("Performing fit and analysis")
    for idx, event in enumerate(Ev_list):
        hits_number.append(event["Hits"])
        if event["Accepted"]:
            selected_ev += 1
            ev = event["Dataframe"]
            chambers = event["Chambers"]
            n_layers = event["Layers"]

            #Local linear fit
            lf_results = Local_Fit(ev, chambers, n_layers, exclusion_layer=True)
            #Global linear fit
            gf_results = Global_Fit_Calibration(ev, chambers, lf_results)
            Ev_residuals.append(gf_results["residuals"])

            # x coordinates for the points in the optimal combination after local fit per chamber
            X_position["up"]   += [x[1] for x in lf_results[0]["optimal_comb"]]
            X_position["down"] += [x[1] for x in lf_results[1]["optimal_comb"]]
            # local fit differences
            lf_diff["slope"]    .append(lf_results[0]["slope"]    -lf_results[1]["slope"])
            lf_diff["intercept"].append(lf_results[0]["intercept"]-lf_results[1]["intercept"])
        # GUI progress bar
        sg.OneLineProgressMeter('File Analyzer', idx+1, tot_ev, 'Analyzing CALIBRATION file', orientation="h", size=(30,30))

    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(hits_number, bins=14, range=(0,70), **hist_kwargs)
    ax.set_title("Hit distribution", fontsize=14)
    ax.set_xlabel("number of hits")
    fig.show()

    fig, (ax1, ax2) = plt.subplots(figsize=(15,5), ncols=2)
    fig.suptitle("X position in local chambers coordinates", fontsize=16)
    ax1.set_title("Up", fontsize=14)
    ax1.hist(X_position["up"]    , bins=30, **hist_kwargs)
    ax1.set_xlabel("x [mm]")
    #
    ax2.set_title("Down", fontsize=14)
    ax2.hist(X_position["down"], bins=30, **hist_kwargs)
    ax2.set_xlabel("x [mm]")
    fig.show()

    fig, (ax1, ax2) = plt.subplots(figsize=(15,5), ncols=2)
    fig.suptitle("Differences between local fits results", fontsize=16)
    ax1.set_title("Slope")
    Gaussian_Fit_Hist(ax1, lf_diff["slope"],     nbins=30, hist_range=(-0.5,0.5), **hist_kwargs)
    ax1.set_xlabel("Slope")
    #
    ax2.set_title("Intercept")
    Gaussian_Fit_Hist(ax2, lf_diff["intercept"], nbins=30, hist_range=(-100,100), **hist_kwargs)
    ax2.set_xlabel("[mm]")
    fig.show()

    Hit_6_layers = 0
    Hit_7_layers = 0
    Hit_8_layers = 0
    Tot_res      = []

    for res in Ev_residuals:
        if len(res) == 0:
            Hit_6_layers += 1
        elif len(res) == 1:
            Hit_7_layers += 1
        elif len(res) == 2:
            Hit_8_layers += 1
        else: continue
        Tot_res += res.tolist()

    fig, (ax1, ax2) = plt.subplots(figsize=(15,5), ncols=2)
    ax1.set_title("Residuals of the points in the excluded layers", fontsize=14)
    Gaussian_Fit_Hist(ax1, Tot_res, nbins= 30, hist_range=(-20,20), **hist_kwargs)
    ax1.set_xlabel("[mm]")
    #
    labels = ['Hit in 8 layers', 'Hit in 7 layers', 'Hit in 6 layers']
    sizes  = [Hit_8_layers/len(Ev_residuals)*100,
              Hit_7_layers/len(Ev_residuals)*100,
              Hit_6_layers/len(Ev_residuals)*100]
    patches, _, _ = ax2.pie(sizes, labels=labels, explode = [.05,.05,.05], autopct='%1.1f%%', shadow=True,
                               wedgeprops={"edgecolor":'black', "linewidth":1.25})
    ax2.legend(patches, labels, loc='upper right', fontsize=14)
    ax2.axis('equal')
    ax2.set_title("Percentage of excluded layers in the fit", fontsize=14)
    fig.show()

    return Ev_residuals, X_position, lf_diff

def Open_File_Physics(data_file):
    with open(data_file) as f:
        tot_ev = len(list(f))

    # loop over events and perform analysis
    Ev_list = []
    selected_ev = 0
    with open(data_file) as f:
        for line in f:
            # read event
            ev, evNum, hits = Read_Data(line)
            # filter by hit position
            ev_left, ev_right, hits_left, hits_right = Points_Filter(ev)
            # select event
            sel_left,  chambers_left,  n_layers_left  = Select_Events_Calibration(ev_left,  hits_left)
            sel_right, chambers_right, n_layers_right = Select_Events_Calibration(ev_right, hits_right)

            if sel_left and sel_right: selected_ev += 1

            # save most important information, in order to plot or reperform the analysis
            # without reading the whole file again
            Ev_list.append(
                {
                    "Number"    : evNum,
                    "Dataframe" : ev,
                    "Hits"      : hits,
                    "Accepted"  : sel_left and sel_right,
                    "Chambers"  : chambers_left+chambers_right,
                    "Layers"    : [n_layers_left, n_layers_right]
                }
            )
            sg.OneLineProgressMeter('File Loader', len(Ev_list), tot_ev, 'key', 'Loading ', orientation="h", size=(30,30))

    print("{:35} = {:d}"     .format("Total number of events in the Run", tot_ev))
    print("{:35} = {:d}"     .format("Number of accepted events"        , selected_ev))
    print("{:35} = {:.4f} %" .format("Fraction of accepted events"      , selected_ev/tot_ev*100))
    # create a pop up for the GUI with summary information
    sg.Popup("{:} = {:d}"    .format("Total number of events in the Run", tot_ev),
             "{:} = {:d}"    .format("Number of accepted events"        , selected_ev),
             "{:} = {:.4f} %".format("Fraction of accepted events"      , selected_ev/tot_ev*100),
            title="Summary", font=font)
    return Ev_list

def Physics(Ev_list):
    tot_ev = len(Ev_list)
    # number of hits selected events
    Ev_hits = {"left":[], "right":[]}
    # angular coefficient of fit
    Ev_slope = {"left":[], "right":[]}
    # residuals from good events fit
    Ev_residuals = {"left":[], "right":[]}

    print("Performing fit and analysis")
    for idx,event in enumerate(Ev_list):
        if event["Accepted"]:
            ev = event["Dataframe"]
            layers = event["Layers"]

            # filter by hit position
            ev_left, ev_right, hits_left, hits_right = Points_Filter(ev)
            # save hit numbers
            Ev_hits["left"] .append(hits_left )
            Ev_hits["right"].append(hits_right)

            # save slope values
            gf_results_left  = Global_Fit_Physics(ev_left , layers[0])
            gf_results_right = Global_Fit_Physics(ev_right, layers[1])
            Ev_slope["left"] .append(gf_results_left ["slope"])
            Ev_slope["right"].append(gf_results_right["slope"])

            # save residuals
            Ev_residuals["left"]  += gf_results_left ["residuals"]
            Ev_residuals["right"] += gf_results_right["residuals"]
        # GUI progress bar
        sg.OneLineProgressMeter('File Analyzer', idx+1, tot_ev, 'Analyzing PHYSICS file', orientation="h", size=(30,30))

    fig, (ax1, ax2) = plt.subplots(figsize=(15,5), ncols=2)
    fig.suptitle("Hit distribution", fontsize=16)
    ax1.hist(Ev_hits["left"] , **hist_kwargs)
    ax1.set_title("Left Side", fontsize=14)
    ax1.set_xlabel("Number of hits")
    ax2.hist(Ev_hits["right"], **hist_kwargs)
    ax2.set_title("Right Side", fontsize=14)
    ax2.set_xlabel("Number of hits")
    fig.show()

    fig, (ax1, ax2) = plt.subplots(figsize=(15,5), ncols=2)
    fig.suptitle("Slope distribution", fontsize=16)
    Gaussian_Fit_Hist(ax1, Ev_slope["left"] , nbins=30, **hist_kwargs)
    ax1.set_title("Left Side", fontsize=14)
    ax1.set_xlabel("Slope")
    Gaussian_Fit_Hist(ax2, Ev_slope["right"], nbins=30, **hist_kwargs)
    ax2.set_title("Right Side", fontsize=14)
    ax2.set_xlabel("Slope")
    fig.show()

    fig, (ax1, ax2) = plt.subplots(figsize=(15,5), ncols=2)
    fig.suptitle("Residuals", fontsize=16)
    Gaussian_Fit_Hist(ax1, Ev_residuals["left"] , nbins=30, hist_range=(-25,25), **hist_kwargs)
    ax1.set_title("Left Side", fontsize=14)
    ax1.set_xlabel("Residuals [mm]")
    Gaussian_Fit_Hist(ax2, Ev_residuals["right"], nbins=30, hist_range=(-25,25), **hist_kwargs)
    ax2.set_title("Right Side", fontsize=14)
    ax2.set_xlabel("Residuals [mm]")
    fig.show()

    return Ev_hits, Ev_slope, Ev_residuals

def Make_Plot_GUI(ev, calibration=False):
    '''Plots of the background and the hits

    Parameters
    ----------
    ev : dict
        dictionary with information about the event; it contains: Number, Dataframe, Hits, Accepted, Chambers, Layers
    calibration : bool
        wether the event is taken during calibration or physics run
    '''

    plt.figure(figsize = (20,10))
    plots = Plot_Events_GUI(ev["Dataframe"], ev["Number"])
    if ev["Accepted"]: Plot_Fit(ev, plots, calibration=calibration)
    return plt.gcf()

def Plot_Events_GUI(dataframe, evNumber):
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
    plots = Plot_Background_GUI()
    plots[0].set_title("Event: {:d}".format(evNumber), {'size':'18'})
    if dataframe.empty == False:
        xL = dataframe["XL_global"]
        xR = dataframe["XR_global"]
        z  = dataframe["Z_global"]
        for image in plots:
            image.plot(xL, z, "bo", markersize=3)
            image.plot(xR, z, "ro", markersize=3)

    return plots

def Plot_Background_GUI():
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
    # different dimensions and positions with respect to the notebook one
    gridsize = (7, 10)
    ax_global = plt.subplot2grid(gridsize, (0, 0), colspan=4, rowspan=7)
    ax_0 = plt.subplot2grid(gridsize, (0, 8), colspan=2, rowspan=3) # top-right
    ax_1 = plt.subplot2grid(gridsize, (4, 8), colspan=2, rowspan=3) # bottom-right
    ax_2 = plt.subplot2grid(gridsize, (0, 5), colspan=2, rowspan=3) # top-left
    ax_3 = plt.subplot2grid(gridsize, (4, 5), colspan=2, rowspan=3) # bottom-left

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
