
V0FinderModule.h:
Kshort-Mesons have a not negligible lifetime. Hence they can decay outside of the beampipe.
However the Energy loss calculation dE/dx of the fit goes back to the interception point.
The V0Finder corrects this shift and performs a seperate Rave Vertex Fit on the Kshorts.
In order to evaluate the efficiency of this process and benchmark it, this Vertex Fit information is written out at reconstruction level into dataobjects (V0extra.h).

kspretrain_steering.py:  
An Analysis with 1 billion Monte Carlo events is set up. The Data has to be categorized, split up, merged, and processed on external servers.

kspretrain.py:
For speedup of Neural Network training, the data is prepared (write out of variables) in parallel jobs.  

analysis_main.py:
After Kshort and B Mesons are reconstructed and selected by the Classifier,
the CP-Violation shall be measured. The Final DataFrame is analyzed via statistics and backgrounds, then the theoretical model is convoluted with the resolution function and a
fit on the deltaT distrubion determines the wanted parameters.
  
analysis_functions.py:  
Functions for statistics, fitting, and calculating the uncertainty.

analysis_myplot.py:
Classes for plotting and fitting.	

