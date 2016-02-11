#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'tkeck','pjaeger'


import copy
import math
import itertools

import numpy
import pandas
import scipy
import scipy.stats
import matplotlib
import matplotlib.artist
import matplotlib.figure
import matplotlib.gridspec
import matplotlib.colors
import matplotlib.patches
import matplotlib.ticker

import b2stat

from basf2 import *

# Do not use standard backend TkAgg, because it is NOT thread-safe
# You will get an RuntimeError: main thread is not in main loop otherwise!
matplotlib.use("svg")


class Plotter(object):
    """
    Base class for all Plotters.
    """

    #: Plots added to the axis so far
    plots = None
    #: Labels of the plots added so far
    labels = None
    #: Minimum x value
    xmin = None
    #: Maximum x value
    xmax = None
    #: Minimum y value
    ymin = None
    #: Maximum y value
    ymax = None
    #: y limit scale
    yscale = 0.0
    #: x limit scale
    xscale = 0.0
    #: figure which is used to draw
    figure = None
    #: Main axis which is used to draw
    axis = None
    #bins
    bins = None

    fill = None

    def __init__(self, figure=None, axis=None):
        """
        Creates a new figure and axis if None is given, sets the default plot parameters
        @param figure default draw figure which is used
        @param axis default draw axis which is used
        """
        B2INFO("Create new figure for class " + str(type(self)))
        if figure is None:
            self.figure = matplotlib.figure.Figure(figsize=(32, 18))
            self.figure.set_tight_layout(False)
        else:
            self.figure = figure

        if axis is None:
            self.axis = self.figure.add_subplot(1, 1, 1)
        else:
            self.axis = axis

        self.plots = []
        self.labels = []
        self.xmin, self.xmax = float(0), float(1)
        self.ymin, self.ymax = float(0), float(1)
        self.yscale = 0.1
        self.xscale = 0.0

        #: Default keyword arguments for plot function
        self.plot_kwargs = None
        #: Default keyword arguments for errorbar function
        self.errorbar_kwargs = None
        #: Default keyword arguments for fill_between function
        self.errorband_kwargs = None

        self.set_plot_options()
        self.set_errorbar_options()
        self.set_errorband_options()

    def add_subplot(self, gridspecs):
        """
        Adds a new subplot to the figure, updates all other axes
        according to the given gridspec
        @param gridspecs gridspecs for all axes including the new one
        """
        for gs, ax in zip(gridspecs[:-1], self.figure.axes):
            ax.set_position(gs.get_position(self.figure))
            ax.set_subplotspec(gs)
        axis = self.figure.add_subplot(gridspecs[-1], sharex=self.axis)
        return axis

    def save(self, filename):
        """
        Save the figure into a file
        @param filename of the file
        """
        B2INFO("Save figure for class " + str(type(self)))
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        canvas = FigureCanvas(self.figure)
        canvas.print_figure(filename, dpi=50)
        # canvas.show()

        return self

    def set_plot_options(self, plot_kwargs={'linestyle': ''}):
        """
        Overrides default plot options for datapoint plot
        @param plot_kwargs keyword arguments for the plot function
        """
        self.plot_kwargs = copy.copy(plot_kwargs)
        return self

    def set_errorbar_options(self, errorbar_kwargs={'fmt': '.', 'elinewidth': 3, 'alpha': 1}):
        """
        Overrides default errorbar options for datapoint errorbars
        @param errorbar_kwargs keyword arguments for the errorbar function
        """
        self.errorbar_kwargs = copy.copy(errorbar_kwargs)
        return self

    def set_errorband_options(self, errorband_kwargs={'alpha': 0.5}):
        """
        Overrides default errorband options for datapoint errorband
        @param errorbar_kwargs keyword arguments for the fill_between function
        """
        self.errorband_kwargs = copy.copy(errorband_kwargs)
        return self

    def _plot_datapoints(self, axis, x, y, xerr=None, yerr=None):
        """
        Plot the given datapoints, with plot, errorbar and make a errorband with fill_between
        @param x coordinates of the data points
        @param y coordinates of the data points
        @param xerr symmetric error on x data points
        @param yerr symmetric error on y data points
        """
        p = e = f = None
        plot_kwargs = copy.copy(self.plot_kwargs)
        errorbar_kwargs = copy.copy(self.errorbar_kwargs)
        errorband_kwargs = copy.copy(self.errorband_kwargs)
        fill = copy.copy(self.fill)

        if plot_kwargs is None or 'color' not in plot_kwargs:
            color = next(axis._get_lines.prop_cycler)
            color = color["color"]
            plot_kwargs['color'] = color

        else:
            color = plot_kwargs['color']
        color = matplotlib.colors.ColorConverter().to_rgb(color)
        patch = matplotlib.patches.Patch(color=color, alpha=0.5)
        patch.get_color = patch.get_facecolor

        if fill:
            axis.fill_between(x,0,y,alpha="0.5",color=color)
            # self.set_errorband_options(None)
            errorband_kwargs = None

        if plot_kwargs is not None:
            p, = axis.plot(x, y, **plot_kwargs)

        if errorbar_kwargs is not None and (xerr is not None or yerr is not None):
            if 'color' not in errorbar_kwargs:
                errorbar_kwargs['color'] = color
            if 'ecolor' not in errorbar_kwargs:
                errorbar_kwargs['ecolor'] = [0.5 * x for x in color]
            e = axis.errorbar(x, y, xerr=xerr, yerr=yerr, **errorbar_kwargs)

        if errorband_kwargs is not None and yerr is not None:
            if 'color' not in errorband_kwargs:
                errorband_kwargs['color'] = color
            x1 = x
            y1 = y - yerr
            y2 = y + yerr
            if xerr is not None:
                boundaries = numpy.r_[numpy.c_[x - xerr, y1, y2], numpy.c_[x + xerr, y1, y2]]
                boundaries = boundaries[boundaries[:, 0].argsort()]
                x1 = boundaries[:, 0]
                y1 = boundaries[:, 1]
                y2 = boundaries[:, 2]
            f = axis.fill_between(x1, y1, y2, interpolate=True, **errorband_kwargs)

        return (patch, p, e, f)

    def add(self, *args, **kwargs):
        """
        Add a new plot to this plotter
        """
        return NotImplemented

    def finish(self, *args, **kwargs):
        """
        Finish plotting and set labels, legends and stuff
        """
        return NotImplemented

    def scale_limits(self):
        """
        Scale limits to increase distance to boundaries
        """
        self.ymin *= 1.0 - math.copysign(self.yscale, self.ymin)
        self.ymax *= 1.0 + math.copysign(self.yscale, self.ymax)
        self.xmin *= 1.0 - math.copysign(self.xscale, self.xmin)
        self.xmax *= 1.0 + math.copysign(self.xscale, self.xmax)
        return self



class PurityOverEfficiency(Plotter):
    """
    Plots the purity over the efficiency also known as ROC curve
    """

    def add(self, data, column, signal_mask, bckgrd_mask, weight_column=None):
        """
        Add a new curve to the ROC plot
        @param data pandas.DataFrame containing all data
        @param column which is used to calculate efficiency and purity for different cuts
        @param signal_mask boolean numpy.array defining which events are signal events
        @param bckgrd_mask boolean numpy.array defining which events are background events
        @param weight_column column in data containing the weights for each event
        """
        hists = b2stat.Histograms(data, column, {'Signal': signal_mask, 'Background': bckgrd_mask}, weight_column=weight_column)
        efficiency, efficiency_error = hists.get_efficiency(['Signal'])
        purity, purity_error = hists.get_purity(['Signal'], ['Background'])

        self.xmin, self.xmax = numpy.nanmin([efficiency.min(), self.xmin]), numpy.nanmax([efficiency.max(), self.xmax])
        self.ymin, self.ymax = numpy.nanmin([numpy.nanmin(purity), self.ymin]), numpy.nanmax([numpy.nanmax(purity), self.ymax])

        p = self._plot_datapoints(self.axis, efficiency, purity, xerr=efficiency_error, yerr=purity_error)
        self.plots.append(p)
        self.labels.append(column)
        return self

    def finish(self):
        """
        Sets limits, title, axis-labels and legend of the plot
        """
        self.axis.set_xlim((self.xmin, self.xmax))
        self.axis.set_ylim((self.ymin, self.ymax))
        self.axis.set_title("ROC Purity Plot")
        self.axis.get_xaxis().set_label_text('Efficiency')
        self.axis.get_yaxis().set_label_text('Purity')
        self.axis.legend([x[0] for x in self.plots], self.labels, loc='best', fancybox=True, framealpha=0.5)
        return self


class RejectionOverEfficiency(Plotter):
    """
    Plots the rejection over the efficiency also known as ROC curve
    """

    def add(self, data, column, signal_mask, bckgrd_mask, weight_column=None):
        """
        Add a new curve to the ROC plot
        @param data pandas.DataFrame containing all data
        @param column which is used to calculate efficiency and purity for different cuts
        @param signal_mask boolean numpy.array defining which events are signal events
        @param bckgrd_mask boolean numpy.array defining which events are background events
        @param weight_column column in data containing the weights for each event
        """
        hists = b2stat.Histograms(data, column, {'Signal': signal_mask, 'Background': bckgrd_mask}, weight_column=weight_column)
        efficiency, efficiency_error = hists.get_efficiency(['Signal'])
        rejection, rejection_error = hists.get_efficiency(['Background'])
        rejection = 1 - rejection

        self.xmin, self.xmax = numpy.nanmin([efficiency.min(), self.xmin]), numpy.nanmax([efficiency.max(), self.xmax])
        self.ymin, self.ymax = numpy.nanmin([rejection.min(), self.ymin]), numpy.nanmax([rejection.max(), self.ymax])

        p = self._plot_datapoints(self.axis, efficiency, rejection, xerr=efficiency_error, yerr=rejection_error)
        self.plots.append(p)
        self.labels.append(column)
        return self

    def finish(self):
        """
        Sets limits, title, axis-labels and legend of the plot
        """
        self.axis.set_xlim((self.xmin, self.xmax))
        self.axis.set_ylim((self.ymin, self.ymax))
        self.axis.set_title("ROC Rejection Plot")
        self.axis.get_xaxis().set_label_text('Signal Efficiency')
        self.axis.get_yaxis().set_label_text('Background Rejection')
        self.axis.legend([x[0] for x in self.plots], self.labels, loc='best', fancybox=True, framealpha=0.5)
        return self


class Diagonal(Plotter):
    """
    Plots the purity in each bin over the classifier output.
    """

    def add(self, data, column, signal_mask, bckgrd_mask, weight_column=None):
        """
        Add a new curve to the Diagonal plot
        @param data pandas.DataFrame containing all data
        @param column which is used to calculate purity for different cuts
        @param signal_mask boolean numpy.array defining which events are signal events
        @param bckgrd_mask boolean numpy.array defining which events are background events
        @param weight_column column in data containing the weights for each event
        """
        hists = b2stat.Histograms(data, column, {'Signal': signal_mask, 'Background': bckgrd_mask}, weight_column=weight_column)
        purity, purity_error = hists.get_purity_per_bin(['Signal'], ['Background'])

        self.xmin, self.xmax = min(hists.bin_centers.min(), self.xmin), max(hists.bin_centers.max(), self.xmax)
        # self.ymin, self.ymax = numpy.nanmin([numpy.nanmin(purity), self.ymin]), numpy.nanmax([numpy.nanmax(purity), self.ymax])
        self.ymin, self.ymax = 0, 1

        p = self._plot_datapoints(self.axis, hists.bin_centers, purity, xerr=hists.bin_widths / 2.0, yerr=purity_error)
        self.plots.append(p)
        self.labels.append(column)
        return self

    def finish(self):
        """
        Sets limits, title, axis-labels and legend of the plot
        """
        self.scale_limits()
        self.axis.plot((0.0, 1.0), (0.0, 1.0), color='black')
        self.axis.set_xlim((self.xmin, self.xmax))
        self.axis.set_ylim((self.ymin, self.ymax))
        self.axis.set_title("Diagonal Plot")
        self.axis.get_xaxis().set_label_text('Classifier Output')
        self.axis.get_yaxis().set_label_text('Purity Per Bin')
        self.axis.legend([x[0] for x in self.plots], self.labels, loc='best', fancybox=True, framealpha=0.5)
        return self


class Distribution(Plotter):
    """
    Plots distribution of a quantity
    """

    popt = None
    hist = None
    hists = None
    title = None
    Data = None
    x0 = None
    bin_width = None
    unit = None

    def __init__(self, figure=None, axis=None, normed_to_all_entries=False, normed_to_bin_width=False, keep_first_binning=False):
        """
        Creates a new figure and axis if None is given, sets the default plot parameters
        @param figure default draw figure which is used
        @param axis default draw axis which is used
        @param normed true if histograms should be normed before drawing
        @param keep_first_binning use the binning of the first distribution for further plots
        """
        super(Distribution, self).__init__(figure, axis)
        #: Normalize histograms before drawing them
        self.normed_to_all_entries = normed_to_all_entries
        self.normed_to_bin_width = normed_to_bin_width
        # if self.normed_to_all_entries or self.normed_to_bin_width:
        self.ymin = float(0)
        self.ymax = float('-inf')
        self.xmin = float('inf')
        self.xmax = float('-inf')

        #: Keep first binning if user wants so
        self.keep_first_binning = keep_first_binning
        #: first binning
        self.first_binning = None

    def add(self, data, column, mask=None, weight_column=None):
        """
        Add a new distribution to the plots
        @param data pandas.DataFrame containing all data
        @param column which is used to calculate distribution histogram
        @param mask boolean numpy.array defining which events are used for the histogram
        @param weight_column column in data containing the weights for each event
        """
        self.Data = data[column]

        if mask is None:
            mask = numpy.ones(len(data)).astype('bool')

        if self.bins is None:
            bins = 100
        else:
            bins=self.bins
        if self.keep_first_binning and self.first_binning is not None:
            bins = self.first_binning
        hists = b2stat.Histograms(data, column, {'Total': mask}, weight_column=weight_column, bins=bins)
        self.hists = hists
        if self.keep_first_binning and self.first_binning is None:
            self.first_binning = hists.bins
        hist, hist_error = hists.get_hist('Total')
        self.hist = hist

        if self.normed_to_all_entries:
            normalization = float(numpy.sum(hist))
            hist = hist / normalization
            hist_error = hist_error / normalization

        if self.normed_to_bin_width:
            hist = hist / hists.bin_widths
            hist_error = hist_error / hists.bin_widths

        self.xmin, self.xmax = min(hists.bin_centers.min(), self.xmin), max(hists.bin_centers.max(), self.xmax)
        self.ymin, self.ymax = numpy.nanmin([hist.min(), self.ymin]), numpy.nanmax([(hist + hist_error).max(), self.ymax])
        self.bin_width = numpy.round((self.xmax - self.xmin) / float(bins),2)

        p = self._plot_datapoints(self.axis, hists.bin_centers, hist, xerr=hists.bin_widths / 2, yerr=hist_error)
        self.plots.append(p)

        self.labels.append(column)

        return self

    def gauss(self,x,A,mu,sigma):
        return  A * numpy.exp(-(x-mu)**2/(2.*sigma**2))

    def triple(self,x,A1,mu1,sigma1,A2,mu2,sigma2,A3,mu3,sigma3):
        return self.gauss(x,A1,mu1,sigma1)+self.gauss(x,A2,mu2,sigma2)+self.gauss(x,A3,mu3,sigma3)

    def performFit(self):
        """
        Fit Distribution with triple Guassian function and validate via Kolmogorov-Smirnov test.
        """
      
        self.fill = False

        self.x0 = [len(self.Data)/2.,0.0,numpy.std(self.Data), \
                   len(self.Data)/3.,0.0,numpy.std(self.Data)*2, \
                   len(self.Data)/4.,0.0,numpy.std(self.Data)*0.5]
        self.popt, pcov = scipy.optimize.curve_fit(self.triple, self.hists.bin_centers, self.hist, p0=self.x0, sigma=None, absolute_sigma=False)
        print("in",self.x0)
        print("out",self.popt)
        sigmaw = self.getSigmaW()
        X = numpy.linspace(self.hists.bin_centers[0],self.hists.bin_centers[-1],1000)
        fithist = numpy.array([self.triple(x,*self.popt) for x in X])

        # Kolmogorov smirnov test
        ks = scipy.stats.ks_2samp(self.hist, fithist)
        props = dict(boxstyle='round', edgecolor='gray', facecolor='white', linewidth=0.1, alpha=0.5)
        self.axis.text(0.6, 0.5, r'$KS-test: p='+str(numpy.round(ks[1],3))+'$', fontsize=20, bbox=props
                       ,verticalalignment='top', horizontalalignment='left', transform=self.axis.transAxes)


        self.axis.text(0.6, 0.6, r'$<\Delta t> ='+str(numpy.round(sigmaw,3))+'ps$', fontsize=20, bbox=props
                       ,verticalalignment='top', horizontalalignment='left', transform=self.axis.transAxes)



        self.set_plot_options( plot_kwargs={ 'marker':' ','linestyle':'-'})
        p = self._plot_datapoints(self.axis, X,fithist, xerr=None , yerr=None)
        self.plots.append(p)
       
        return self




    def getSigmaW(self):
	"""
	Get weighted Uncertainty of the tiple Gauss fit.
	"""
	
        result1 = scipy.integrate.quad(lambda x: self.gauss(x,*self.popt[:3]),-0.5,0.5)
        result2 = scipy.integrate.quad(lambda x: self.gauss(x,*self.popt[3:6]),-0.5,0.5)
        result3 = scipy.integrate.quad(lambda x: self.gauss(x,*self.popt[6:]),-0.5,0.5)
        result = scipy.integrate.quad(lambda x: self.triple(x,*self.popt),-0.5,0.5)
        sigma1, sigma2,sigma3 = abs(self.popt[2]),abs(self.popt[5]),abs(self.popt[8])
        sigmaw = (sigma1*result1[0] +sigma2*result2[0]+sigma3*result3[0]) \
           / float(result[0])
        print("SIGMAS",sigma1,sigma2,sigma3,sigmaw)
        return sigmaw

    def getPopt(self):
        return self.popt

 

    def finish(self):
        """
        Sets limits, title, axis-labels and legend of the plot
        """
        self.scale_limits()
        self.axis.set_xlim((self.xmin, self.xmax))
        self.axis.set_ylim((self.ymin, self.ymax))
        self.axis.set_title("Distribution Plot")
        self.axis.get_xaxis().set_label_text('Classifier Output')
        if self.normed_to_all_entries and self.normed_to_bin_width:
            self.axis.get_yaxis().set_label_text('# Entries per Bin / (# Entries * Bin Width)')
        elif self.normed_to_all_entries:
            self.axis.get_yaxis().set_label_text('# Entries per Bin / # Entries')
        elif self.normed_to_bin_width:
            self.axis.get_yaxis().set_label_text('# Entries per Bin / Bin Width')
        elif self.unit is not None:
            self.axis.get_yaxis().set_label_text('# entries /' + str(self.bin_width)+str(self.unit))
            self.axis.legend([x[0] for x in self.plots], self.labels, loc='best', fancybox=True, framealpha=0.5)

        else:
            self.axis.get_yaxis().set_label_text('# Entries per Bin')
            self.axis.legend([x[0] for x in self.plots], self.labels, loc='best', fancybox=True, framealpha=0.5)
        return self


class Asymmetry(Plotter):
    """
    Plots the difference between two histograms
    """

    def add(self, data, column, minuend_mask, subtrahend_mask, weight_column=None):
        """
        Add a new difference plot
        @param data pandas.DataFrame containing all data
        @param column which is used to calculate distribution histogram
        @param minuend_mask boolean numpy.array defining which events are for the minuend histogram
        @param subtrahend_mask boolean numpy.array defining which events are for the subtrahend histogram
        @param weight_column column in data containing the weights for each event
        """

        hists = b2stat.Histograms(data, column,
                                  {'Minuend': minuend_mask, 'Subtrahend': subtrahend_mask}, weight_column=weight_column,bins=self.bins)
        minuend, minuend_error = hists.get_hist('Minuend')
        subtrahend, subtrahend_error = hists.get_hist('Subtrahend')
        difference, difference_error = (minuend - subtrahend) /(minuend+subtrahend).astype(float), b2stat.poisson_error(minuend + subtrahend)

        difference_error = numpy.sqrt( (2*subtrahend/((minuend+subtrahend)**2) * b2stat.poisson_error(minuend)  )**2  +\
                                       (2*minuend/((minuend+subtrahend)**2) * b2stat.poisson_error(subtrahend)  )**2  )

        self.xmin, self.xmax = min(hists.bin_centers.min(), self.xmin), max(hists.bin_centers.max(), self.xmax)
        self.ymin = min((difference - difference_error).min(), self.ymin)
        self.ymax = max((difference + difference_error).max(), self.ymax)

        p = self._plot_datapoints(self.axis, hists.bin_centers, difference, xerr=hists.bin_widths / 2, yerr=difference_error)
        self.plots.append(p)
        self.labels.append(column)
        return self

    def finish(self, line_color='black'):
        """
        Sets limits, title, axis-labels and legend of the plot
        """
        self.axis.plot((self.xmin, self.xmax), (0, 0), color=line_color, linewidth=4)
        self.scale_limits()
        self.axis.set_xlim((self.xmin, self.xmax))
        self.axis.set_ylim((self.ymin, self.ymax))
        self.axis.set_title("Asymmetry Plot")
        self.axis.get_yaxis().set_major_locator(matplotlib.ticker.MaxNLocator(5))
        self.axis.get_xaxis().set_label_text(r'$\Delta t [ps]$')
        self.axis.get_yaxis().set_label_text('Asymmetry')
        self.axis.legend([x[0] for x in self.plots], self.labels, loc='best', fancybox=True, framealpha=0.7)
        return self

