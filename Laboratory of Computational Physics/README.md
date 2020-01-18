# Final Project (Group 6) - Notes

## Data format

The data files are organized as a series of events (rows) where each event consist of the coordinates of the hits recorded in that event (N.B.: as the number of hits for even is not fixed, the dataset cannot be seen as a unique big table with a defined number of "columns"). More specifically every raw consists of:

- event number
- number of recorded hits (N)
- hit 1 coordinates
- ...
- hit N coordinates

where N could be zero too and the hits coordinates are:

- *chamber*, which of the four detector registered the hit (numbering is in the range $0-4$)
- *layer*, which layer of that detector
- *xleft*, the x position of the hit **in the local detector coordinates** assuming the track passed at the left of the wire
- *xright*, the x position of the hit **in the local detector coordinates** assuming the track passed at the right of the wire
- *time*, the drift time (redundant)

The local detector coordinates are defined with respect to one side (the left one) of the detector. All the detectors however were placed rotated by 180 degrees, i.e. the x axes of the local and global coordinates have different orientation.

### Notes on data format

- for each recordered hit we have 5 coordinates; this means that the length of a row is **2 + 5N**
- the chambers are numbered in this way:
  - `chamber[0]`: top-right
  - `chamber[1]`: bottom-right
  - `chamber[2]`: top-left
  - `chamber[3]`: bottom-left
- layers are numbered from bottom (`layer = 1`) to top (`layer = 4`)
- assuming each row is a python list (i.e. `list[0] = event number`, `list[1] = N`, ...) and $i \in [0, 1, ..., N-1]$:
  - *chamber*: `list[2+5i]`
  - *layer*: `list[3+5i]`
  - *xleft*: `list[4+5i]`
  - *xright*: `list[5+5i]`
  - *time*: `list[6+5i]`
  - the possible layout for a Pandas `DataFrame` could be:

| Ev_Number | ($n \in [1, N]$) | chamber | layer | XL_local | XR_local | Z_local | time | XL_global | XR_global | Z_global |
| --------- | ---------------- | ------- | ----- | -------- | -------- | ------- | ---- | --------- | --------- | -------- |

~~The hit number  is used as the index of the Dataframe.~~

## Coordinates transformation

The relevant dimensions and the overall coordinate transformations in $z$ and $x$ are given below. Note that all the quantities are expressed in $mm$.

```python
# Cell dimensions
XCELL = 42.
ZCELL = 13.

# X coordinates translation
global_x_shifts = [994.2, 947.4,-267.4,-261.5,]

# Z coordinates translations
local_z_shifts = [z*ZCELL for z in range(0,4)]
global_z_shifts = [823.5, 0, 823.5, 0]
```

### Notes on the transformation

```python
# '-' sign is due to 180° degrees rotation of detectors
x_global =  global_x_shifts[chamber] - x_local

z_local = local_z_shifts[layer]
z_global = global_z_shifts[chamber] + z_local
```

## Functions

### Read data

- **Input**: 1 row of data_file.txt, e.g. 1 event, passed as a `list`
- **Output**: Pandas `DataFrame` as in the *Data Format* paragraph, Number of the Event, Number of hits in the Event

This function takes in input one event at time, and then outputs a Pandas `DataFrame` as described in the previous section. In addition, the transformation from local to global coordinates is performed.

### Plot background

- **Input**: `None`
- **Output**: `list`[pyplot `Axes`] (global image + 4 detectors zooms)

Five plots are given as output: one image of the whole detector, and one for each of the 4 chambers. This function is made in order to do the plot of the background only one time, instead of doing that for every Event.

In order to plot the `Axes` in the `list`, and get a good layout:
```python
gridsize = (5, 2)
fig = plt.figure(figsize = (12, 24))
axes = plot_background()
plt.show()
```

### Plot events

- **Input**: Pandas `DataFrame` (1 event) + event number
- **Output**: `list`\[pyplot `Axes`\] (global image + 4 detectors zooms)

The input of the function is the Pandas `DataFrame` made by the *Read Data* function, and the event number (this is due to the fact that, if the are no hits, the `DataFrame` is empty, and therefore we can't get the Event number from that). Five plots are given as output: one image of the whole detector, and one for each of the 4 chambers. In the images there will be the points of the hits tracked in the event (left/right positions must have different colors).

### Select Events (Calibration)

- **Input**: Pandas `DataFrame` (1 event)
- **Output**: True/False

The input of the function is the Pandas `DataFrame` made by the *Read Data* function. The output is a boolean value, which labels the good calibration events.

**We need to plot the histogram of the frequency of the number of hits, in order to find out the best requirements for good events.**

*Possible choice (to be evalueted)*: Good events requires at least 6 hits (in different layers) either in the left or in the right side of the detector.

### Local linear fit

- **Input**: Pandas `DataFrame` (1 event)
- **Output**: [[slope, intercept] for each chamber]

The input of the function is the Pandas `DataFrame` made by the *Read Data* function. The output is a list of list with the coefficients of the linear regression (e.g. `scipy.stats.linregress`) for each chamber.

The fit is only made for good events, which means the return of *Select Events (Calibration)* function is `True`. If there are no hits in the chamber, the list returned should be `[False, False]`.

The fit has to be made considering all of the possible permutation of the left/right signals; the result will be chosen by selecting the fit with the lowest $\chi^2$.

### Global linear fit

- **Input**: Pandas `DataFrame` (1 event)
- **Output**: [[slope, intercept] for each side (left/right)]

The input of the function is the Pandas `DataFrame` made by the *Read Data* function. The output is a list of list with the coefficients of the linear regression (e.g. `scipy.stats.linregress`) for each side.

The fit is only made for good events, which means the return of *Select Events (Calibration)* function is `True`, for calibration runs, or *Select Events (Physics)* function is `True`, for physics runs. If there are no hits in one of the sides (calibration runs), the list returned should be `[False, False]`.

The fit has to be made considering all of the possible permutation of the left/right signals; the result will be chosen by selecting the fit with the lowest $\chi^2$.

**This part is a draft, and should be analyze better**
> The fit should be made using a fixed number of points (e.g. 6 in different layers), even if we have more. If we have more signals, they should be used to check the goodness of the fit (compare the measured position with ones calculated with fit results).

### Plot events with fit

- **Input**: pyplot figure + fit parameters
- **Output**: pyplot Figure (global image)

The input of the function is the image of the global detector made by the *Plot Events* function, and the lists of parameters retrieved from *Local linea fit* and *Global linear fit*. The plot of the whole detector with hits and linear regressions is given as output; global fit will be dislayed with solid lines, while local one with dashed lines.