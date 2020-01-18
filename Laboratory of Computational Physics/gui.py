import PySimpleGUI as sg
#sg.theme("Topanga")
# set font size
font = ("bitstream charter", 14, "")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from Functions_GUI import *

# magic function to get a tkinter object from a pyplot Figure to display in the GUI
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

##############
# GUI LAYOUT #
##############

# listbox where to display the events recorded in the file
listbox_widget = sg.Listbox(values=[], select_mode='LISTBOX_SELECT_MODE_SINGLE', enable_events=True, size=(10, 30), bind_return_key=True, disabled=True, key='_EventList_', font=font)

# get the dimension for the canvas where to display the plot
fig = plt.figure(figsize = (20,10))
figure_x, figure_y, figure_w, figure_h = fig.bbox.bounds
# canvs where to display the events
canvas_widget  = sg.Canvas(size=(figure_w, figure_h), key='_Plot_')

# checkbox to select if calibration or physics run
checkbox_widget = [sg.Radio("Calibration", "RADIO", default=True, auto_size_text=True, key="_Calibration_", font=font), sg.Radio("Physics", "RADIO", default=False, auto_size_text=True, font=font)]

# group listbox and a button to perform the analyses
column_widget = sg.Column([[listbox_widget], [sg.Submit('Analyze', key="_Analyze_", font=font, disabled=True)]])

# build the layout of the GUI
# (each list describes one row)
layout = [
    [sg.Text('Select File', font=font), sg.InputText(key='_InputFileName_', font=font), sg.FileBrowse(key='_InputFile_', font=font),
     sg.Submit('Load Datafile', key='_Load_', font=font), checkbox_widget[0], checkbox_widget[1]],
     [column_widget, canvas_widget],
         ]

# create the GUI window
window = sg.Window('Event Display', layout, resizable=True, finalize=True)
# get GUI elements
list_box = window['_EventList_']
plot_el = window['_Plot_']

# variable used to dispaly the plot in the canvas
figure_agg = None
# datafile name
data_file = ""
# list of events in the datafile
Ev_list = []
# indeces of accepted events
highlight = []

# event loop to process the events
while True:
    # read returns from the windows
    event, values = window.read()

    ### get window events and handle them

    # close window
    if event in (None, 'Exit'):
        break

    # load datafile if the button is pressed and the datafile is specified
    elif event == "_Load_" and values['_InputFileName_']:
        data_file = values['_InputFileName_']
        # get the calibration flag from the checkbox and call the right function
        if values['_Calibration_'] :
            Ev_list = Open_File_Calibration(data_file)
        else :
            Ev_list = Open_File_Physics(data_file)
        # once the analysis is done, update the list of events in the listbox
        # and highlight the accepted events
        highlight = [i for i in range(len(Ev_list)) if Ev_list[i]["Accepted"]]
        list_box.Update(values=[str(i) for i in range(len(Ev_list))], disabled=False, set_to_index=highlight)
        # update the Analyze button to perform the analysis since the datafile has been loaded
        window["_Analyze_"].Update(disabled=False)

    # plot selected event on click over the listbox
    elif event == "_EventList_":
        # ** IMPORTANT ** Clean up previous drawing before drawing again
        if figure_agg: figure_agg.get_tk_widget().forget()
        # get the index of the events from the listbox
        n = int(values['_EventList_'][0])
        # make the plot and display it in the canvas
        fig = Make_Plot_GUI(Ev_list[n], values['_Calibration_'])
        figure_agg = draw_figure(plot_el.TKCanvas, fig)  # draw the figure
        # need to update again the listbox in order to have accepted events highlighted
        list_box.Update(set_to_index=highlight)

    # perform the analyses (plot will pop up in pyplot windows)
    elif event == "_Analyze_" and Ev_list:
        if values['_Calibration_'] :
            Calibration(Ev_list)
        else:
            Physics(Ev_list)

# clode the window at the end of the execution
window.close()
