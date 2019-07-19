import SNIDsn
import SNIDdataset as snid

import numpy as np
import scipy

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_color_codes('colorblind')
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator


import plotly.plotly as ply
import plotly.graph_objs as go
import plotly.tools as tls

import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import distance
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

import pickle

from scipy.io.idl import readsav
import pylab as pl


def readtemplate(tp):
    """
    Quick function for reading in meanspec templates.

    Parameters
    ----------
    tp : string
        SN type

    Returns
    -------
    s : meanspec .sav file.

    """
    if tp=='IcBL':
        s = readsav('PCvsTemplates/meanspec%s_1specperSN_15_ft.sav'%tp)
    else:
        s = readsav('PCvsTemplates/meanspec%s_1specperSN_15.sav'%tp)

    return s

def plotPCs(s, tp, c, ax, eig, ewav, sgn):
    lines = []
    for i,e in enumerate(eig):
        line = ax.plot(ewav, sgn[i]*2*e +5-1.0*i, label="PCA%i"%i)
        lines.append(line)
        if i:
            ax.fill_between(s.wlog, s.fmean + s.fsdev+ 5-1.0*i,
                            s.fmean - s.fsdev +5-1.0*i, 
                    color = c, alpha = 0.1)
        else:
            ax.fill_between(s.wlog, s.fmean + s.fsdev +5-1.0*i,
                            s.fmean - s.fsdev +5-1.0*i, 
                    color = c, alpha = 0.1, label=tp+' Template')
            
    ax.set_xlim(4000,7000)
    ax.set_xlabel("wavelength ($\AA$)",fontsize=26)
    ax.set_ylim(0, 8)
    return ax, lines
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 3, x.max() + 3
    y_min, y_max = y.min() - 3, y.max() + 3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, alphasvm):
    """
    Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax : matplotlib axes object
    clf : a classifier
    xx : meshgrid ndarray
    yy : meshgrid ndarray
    alphasvm : float
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    IIbZ = len(Z[Z==1])
    IbZ = len(Z[Z==2])
    IcZ = len(Z[Z==3])
    IcBLZ = len(Z[Z==4])
    colors = []
    if IIbZ != 0:
        colors.append(self.IIb_color)
    if IbZ != 0:
        colors.append(self.Ib_color)
    if IcZ != 0:
        colors.append(self.Ic_color)
    if IcBLZ != 0:
        colors.append(self.IcBL_color)
    nbins = len(colors)
    cmap_name = 'mymap'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=nbins)
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, cmap=cm, alpha=alphasvm)
    return out


class SNePCA:

    def __init__(self, snidset, phasemin, phasemax):
        self.snidset = snidset
        self.phasemin = phasemin
        self.phasemax = phasemax

        self.IIb_color = 'g'
        self.Ib_color = 'mediumorchid'
        self.Ic_color = 'r'
        self.IcBL_color = 'gray'
        self.H_color = 'steelblue'
        self.He_color = 'indianred'

        self.IIb_ellipse_color = 'g'
        self.Ib_ellipse_color = 'mediumorchid'
        self.Ic_ellipse_color = 'r'
        self.IcBL_ellipse_color = 'gray'
        
        nspec = snid.numSpec(self.snidset)
        snnames = list(self.snidset.keys())
        tmpobj = self.snidset[snnames[0]]
        nwvlbins = len(tmpobj.wavelengths)
        self.wavelengths = tmpobj.wavelengths

        specMatrix = np.ndarray((nspec, nwvlbins))
        pcaNames = []
        pcaPhases = []
        count = 0
        for snname in snnames:
            snobj = self.snidset[snname]
            phasekeys = snobj.getSNCols()
            for phk in phasekeys:
                specMatrix[count,:] = snobj.data[phk]
                count = count + 1
                pcaNames.append(snname)
                pcaPhases.append(phk)
        self.pcaNames = np.array(pcaNames)
        self.pcaPhases = np.array(pcaPhases)
        self.specMatrix = specMatrix

        return

    def getSNeNameMask(self, excludeSNe):
        """
        Returns a mask to filter out the named SNe in excludeSNe.

        Parameters
        ----------
        excludeSNe : list
            list of strings that are SN names.

        Returns
        -------
        nameMask : np.array

        """
        allNames = list(self.snidset.keys())
        nameMask = np.logical_not(np.isin(allNames, excludeSNe))
        return nameMask

    def getSNeTypeMasks(self):
        """
        Returns
        -------
        Masks that select each of the 4 major SESN types.

        """
        snnames = list(self.snidset.keys())
        snnames = self.pcaNames
        typeinfo = snid.datasetTypeDict(self.snidset)
        IIblist = typeinfo['IIb']
        Iblist = typeinfo['Ib']
        Iclist = typeinfo['Ic']
        IcBLlist = typeinfo['IcBL']

        IIbmask = np.in1d(snnames, IIblist)
        Ibmask = np.in1d(snnames, Iblist)
        Icmask = np.in1d(snnames, Iclist)
        IcBLmask = np.in1d(snnames, IcBLlist)

        return IIbmask, Ibmask, Icmask, IcBLmask


    def snidPCA(self):
        """
        Calculates PCA eigenspectra and stores them in self.evecs
        Returns
        -------

        """
        pca = PCA()
        pca.fit(self.specMatrix)
        self.evecs = pca.components_
        self.evals = pca.explained_variance_ratio_
        self.evals_cs = self.evals.cumsum()
        return

    def calcPCACoeffs(self):
        """
        Calculates the pca coefficients for all spectra and stores
        them in self.pcaCoeffMatrix
        Returns
        -------

        """
        self.pcaCoeffMatrix = np.dot(self.evecs, self.specMatrix.T).T

        for i, snname in enumerate(list(self.snidset.keys())):
            snobj = self.snidset[snname]
            snobj.pcaCoeffs = self.pcaCoeffMatrix[i,:]
        return






    def reconstructSpectrumGrid(self, figsize, snname, phasekey,
                                Nhostgrid, nPCAComponents, fontsize,
                                leg_fontsize, ylim=(-2,2), dytick=1):
        """
        Reconstructs the spectrum of snname at phase phasekey.

        Parameters
        ----------
        figsize : tuple
        snname : string
            name of SN
        phasekey : string
            phase of spectrum
        Nhostgrid : int
            Number of subgrids
        nPCAComponents : list
            list of options for number of
            eigenspectra to include in reconstruction.
        fontsize : int
        leg_fontsize : int
            legend fontsize
        ylim : tuple
        dytick : int

        Returns
        -------
        f : plt.figure
        hostgrid : GridSpec

        """
        
        f = plt.figure(figsize=figsize)
        hostgrid = gridspec.GridSpec(Nhostgrid,1)
        hostgrid.update(hspace=0.2)

        subgrid = gridspec.GridSpecFromSubplotSpec(len(nPCAComponents), 1, subplot_spec=hostgrid[0:,0], hspace=0)

        snobj = self.snidset[snname]
        datasetMean = np.mean(self.specMatrix, axis=0)
        trueSpec = snobj.data[phasekey]
        pcaCoeff = np.dot(self.evecs, (trueSpec - datasetMean))
        plt.tick_params(axis='both', which='both', bottom='off', top='off',\
                            labelbottom='off', labelsize=40, right='off', left='off', labelleft='off')
        f.subplots_adjust(hspace=0, top=0.95, bottom=0.1, left=0.12, right=0.93)
        
        for i, n in enumerate(nPCAComponents):
            ax = plt.subplot(subgrid[i,0])
            ax.plot(snobj.wavelengths, trueSpec, c='k', linewidth=4.0, alpha=0.5,label=snname+' True Spectrum')
            ax.plot(snobj.wavelengths, datasetMean + (np.dot(pcaCoeff[:n], self.evecs[:n])), c='b', linestyle='--', linewidth=4.0,label=snname + ' Reconstruction')
            ax.tick_params(axis='both',which='both',labelsize=20)
            if i == 0:
                ax.legend(loc='lower left', fontsize=leg_fontsize)
            if i < len(nPCAComponents) - 1:
                plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='on',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off') # labels along the bottom edge are off
            ax.set_ylim(ylim)
            yticks = np.arange(ylim[0] - np.sign(ylim[0])*dytick, ylim[-1], dytick)
            ax.set_yticks(yticks)
            ax.set_yticklabels([])
            ax.tick_params(axis='y', length=20, direction="inout")

            if i == 0:
                # Balmer lines
                trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                trans2 = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)


                ax.axvspan(6213, 6366, alpha=0.1, color=self.H_color) #H alpha -9000 km/s to -16000 km/s
                s = r'$\alpha$'
                xcord = (6213+6366)/2.0
                ax.text(xcord, 1.05, 'H'+s, fontsize=fontsize, horizontalalignment='center',\
                        verticalalignment='center',transform=trans)
                ax.axvspan(4602, 4715, alpha=0.1, color=self.H_color) #H Beta -9000 km/s to-16000 km/s
                s = r'$\beta$'
                xcord = (4602+4715)/2.0
                ax.text(xcord, 1.05, 'H'+s, fontsize=fontsize, horizontalalignment='center',\
                        verticalalignment='center',transform=trans)
                #HeI 5876, 6678, 7065
                ax.axvspan(5621, 5758, alpha=0.1, color=self.He_color) #HeI5876 -6000 km/s to -13000 km/s
                ax.text((5621+5758)/2.0, 1.05, 'HeI', fontsize=fontsize, horizontalalignment='center',\
                        verticalalignment='center', transform=trans)
                ax.axvspan(6388, 6544, alpha=0.1, color=self.He_color)
                ax.text((6388+6544)/2.0, 1.05, 'HeI', fontsize=fontsize, horizontalalignment='center',\
                        verticalalignment='center', transform=trans)
                ax.axvspan(6729, 6924, alpha=0.1, color=self.He_color)
                ax.text((6729+6924)/2.0, 1.05, 'HeI', fontsize=fontsize, horizontalalignment='center',\
                        verticalalignment='center', transform=trans)
 



            if i > 0:
                # Balmer lines
                trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                ax.axvspan(6213, 6366, alpha=0.1, color=self.H_color) #H alpha -9000 km/s to -16000 km/s
                ax.axvspan(4602, 4715, alpha=0.1, color=self.H_color) #H Beta -9000 km/s to-16000 km/s
                ax.axvspan(5621, 5758, alpha=0.1, color=self.He_color) #HeI5876 -6000 km/s to -13000 km/s
                ax.axvspan(6388, 6544, alpha=0.1, color=self.He_color)
                ax.axvspan(6729, 6924, alpha=0.1, color=self.He_color)
            if n == 0:
                text = 'mean'
            elif n == 1:
                text = "1 component\n"
                text += r"$(\sigma^2_{tot} = %.2f)$" % self.evals_cs[n - 1]

            else:
                text = "%i components\n" % n
                text += r"$(\sigma^2_{tot} = %.2f)$" % self.evals_cs[n - 1]
                #text = '(n PCA = %i,$\sigma^{2}$ =  %.0f'%(n, 100*self.evals_cs[n-1])+'%)'
            ax.text(0.75, 0.3, 'nPC = %i'%(n), fontsize=fontsize, ha='left', va='top', transform=ax.transAxes)
            reconstruct_percent = np.sum(np.abs(pcaCoeff[:n]))/np.sum(np.abs(pcaCoeff))
            text = '$\sigma^{2}$ = %.2f'%(100*reconstruct_percent)+'%'
            ax.text(0.75, 0.15, text, fontsize=fontsize,ha='left', va='top', transform=ax.transAxes)
            f.axes[-1].set_xlabel(r'${\rm Wavelength\ (\AA)}$',fontsize=fontsize)
            f.axes[-1].tick_params(axis='x', length=30, direction="inout", labelsize=fontsize-10)
            f.axes[-1].tick_params(axis='x', which='minor', length=15, direction='inout')
            f.text(0.055, 1.0/2.0, 'Relative Flux', verticalalignment='center', rotation='vertical', fontsize=fontsize)


        return f, hostgrid


    def pcaCumPlot(self, figsize, fontsize):
        """
        Plots cumulative pca percent of variance captured in sample
        as a function of number of eigenspectra.

        Parameters
        ----------
        figsize : tuple
        fontsize : int

        Returns
        -------
        f : plt.figure
        ax : figure axis

        """

        f = plt.figure(figsize=figsize)
        ax = plt.gca()
        xcumsum = np.arange(len(self.evals_cs)+1)
        ycumsum = np.hstack((np.array([0]), self.evals_cs))
        plt.plot(xcumsum, ycumsum, linewidth=4.0,c='k')
        plt.scatter(xcumsum, ycumsum, s=150, c='r')
        ax.text(5,self.evals_cs[4]-.075,'(5,%.2f)'%(self.evals_cs[4]),fontsize=60,color='r')
        ax.set_ylabel('Cumulative '+'$\sigma^{2}$', fontsize=65)
        ax.set_xlabel('nPC', fontsize=fontsize)
        ax.tick_params(axis='both',which='both',labelsize=fontsize-10)
        ax.tick_params(axis='x', length=30, direction='inout')
        ax.tick_params(axis='x', which='minor', length=15, direction='inout')
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        return f, ax



    def plotEigenspectra(self, figsize, nshow, ylim=None, fontsize=16):
        """
        Plots the first nshow eigenspectra.

        Parameters
        ----------
        figsize : tuple
        nshow : int
        ylim : tuple
        fontsize : int

        Returns
        -------
        f : plt.figure
        hostgrid : GridSpec

        """
        f = plt.figure(figsize=figsize)
        hostgrid = gridspec.GridSpec(3,1)
        hostgrid.update(hspace=0.2)

        eiggrid = gridspec.GridSpecFromSubplotSpec(nshow, 1, subplot_spec=hostgrid[:2,0], hspace=0)

        for i, ev in enumerate(self.evecs[:nshow]):
            ax = plt.subplot(eiggrid[i,0])
            ax.plot(self.wavelengths, ev, color=self.IcBL_color)

            trans2 = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
            ax.text(0.02,0.85, "(PCA%d, %.0f"%(i+1, 100*self.evals_cs[i])+'%)', horizontalalignment='left',\
                    verticalalignment='center', fontsize=fontsize, transform=trans2)
            ax.tick_params(axis='both',which='both',labelsize=fontsize)
            if not ylim is None:
                ax.set_ylim(ylim)
            if i > -1:
                yticks = ax.yaxis.get_major_ticks()
                yticks[-1].set_visible(False)

            if i == 0:
                # Balmer lines
                trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                trans2 = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)

                ax.text(0.02,1.05, "(PCA#, Cum. Var.)", fontsize=fontsize, horizontalalignment='left',\
                        verticalalignment='center', transform=trans2)

                ax.axvspan(6213, 6366, alpha=0.1, color=self.H_color) #H alpha -9000 km/s to -16000 km/s
                s = r'$\alpha$'
                xcord = (6213+6366)/2.0
                ax.text(xcord, 1.05, 'H'+s, fontsize=fontsize, horizontalalignment='center',\
                        verticalalignment='center',transform=trans)
                ax.axvspan(4602, 4715, alpha=0.1, color=self.H_color) #H Beta -9000 km/s to-16000 km/s
                s = r'$\beta$'
                xcord = (4602+4715)/2.0
                ax.text(xcord, 1.05, 'H'+s, fontsize=fontsize, horizontalalignment='center',\
                        verticalalignment='center',transform=trans)


                ax.axvspan(5621, 5758, alpha=0.1, color=self.He_color) #HeI5876 -6000 km/s to -13000 km/s
                ax.text((5621+5758)/2.0, 1.05, 'HeI5876', fontsize=fontsize, horizontalalignment='center',\
                        verticalalignment='center', transform=trans)
                ax.axvspan(6388, 6544, alpha=0.1, color=self.He_color)
                ax.text((6388+6544)/2.0, 1.05, 'HeI6678', fontsize=fontsize, horizontalalignment='center',\
                        verticalalignment='center', transform=trans)
                ax.axvspan(6729, 6924, alpha=0.1, color=self.He_color)
                ax.text((6729+6924)/2.0, 1.05, 'HeI7065', fontsize=fontsize, horizontalalignment='center',\
                        verticalalignment='center', transform=trans)
            if i > 0:
                # Balmer lines
                trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                ax.axvspan(6213, 6366, alpha=0.1, color=self.H_color) #H alpha -9000 km/s to -16000 km/s
                ax.axvspan(4602, 4715, alpha=0.1, color=self.H_color) #H Beta -9000 km/s to-16000 km/s
                ax.axvspan(5621, 5758, alpha=0.1, color=self.He_color) #HeI5876 -6000 km/s to -13000 km/s
                ax.axvspan(6388, 6544, alpha=0.1, color=self.He_color)
                ax.axvspan(6729, 6924, alpha=0.1, color=self.He_color)


            if i == nshow - 1:
                ax.set_xlabel("Wavelength", fontsize=fontsize)

        ax = plt.subplot(hostgrid[-1])
        ax.boxplot(self.pcaCoeffMatrix)
        ax.set_xlabel('PCA Component #', fontsize=fontsize)
        ax.set_ylabel('PCA Coefficient Value', fontsize=fontsize)
        ax.tick_params(axis='both', which='both', labelsize=fontsize)
        ax.axhline(y=0, color=self.Ic_color)
        xticklabels = ax.xaxis.get_majorticklabels()
        xticklabels[0].set_visible
        for i, tick in enumerate(xticklabels):
            if i%4 != 0:
                tick.set_visible(False)

        return f, hostgrid



    def meanTemplateEig(self, figsize):
        """
        Plots the meanspectra and first 5 eigenspectra.

        Parameters
        ----------
        figsize : tuple

        Returns
        -------
        f : plt.figure
        axs : figure axes

        """
        snIb = readtemplate('Ib')
        snIc = readtemplate('Ic')
        snIIb = readtemplate('IIb')
        snIcBL = readtemplate('IcBL')
        f, axs = plt.subplots(2,2,figsize=figsize, sharex=True, sharey=True)
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        axs[0,0], _ = plotPCs(snIIb, 'IIb','g', axs[0,0], self.evecs[0:5], self.wavelengths,[1,-1,-1,1,-1])
        axs[0,1], _ = plotPCs(snIb, 'Ib','mediumorchid', axs[0,1], self.evecs[0:5], self.wavelengths,[1,-1,-1,1,-1])
        axs[1,0], _ = plotPCs(snIcBL, 'IcBL','k', axs[1,0], self.evecs[0:5], self.wavelengths,[1,-1,-1,1,-1])
        axs[1,1], lines = plotPCs(snIc, 'Ic','r', axs[1,1], self.evecs[0:5], self.wavelengths,[1,-1,-1,1,-1])
        return f, axs



    def pcaPlot(self, pcax, pcay, figsize, alphamean, alphaell, alphasvm,
                purity=False, excludeSNe=[], std_rad=None, svm=False,
                fig=None, ax=None, count=1, svmsc=[], ncv=10, markOutliers=False):
        """

        Parameters
        ----------
        pcax : int
            Eigenspectrum number for x axis
        pcay : int
            Eigenspectrum number of y axis
        figsize : tuple
        alphamean : float
            ALpha for centroid marker
        alphaell : float
            Alpha for ellipses
        alphasvm : float
            Alpha for SVM regions
        purity : Boolean
            Calculates purity of regions if True
        excludeSNe : list
            List of SNe not to include in ellipse calculation
        std_rad : float
            putiry within std_rad number of radii
        svm : Boolean
            Plots SVM regions if True
        fig : plt.figure
        ax : figure axis
        count : int
        svmsc : list
            SVM scores for each CV iteration of SVM
        ncv : int
            Number of cross validation runs
        markOutliers : Boolean
            Marks outliers if True

        Returns
        -------
        f : plt.figure
        svmsc : list
        avgsc : float
            average CV SVM score
        stdsc : float
            CV SVM score standard deviation

        """
        if fig is None:
            f = plt.figure(figsize=figsize)
        else:
            f = fig
        if ax is None:
            ax = plt.gca()
        else:
            ax = ax
        Ic_patch = mpatches.Patch(color=self.Ic_color, label='Ic')
        Ib_patch = mpatches.Patch(color=self.Ib_color, label='Ib')
        IcBL_patch = mpatches.Patch(color=self.IcBL_color, label='IcBL')
        IIb_patch = mpatches.Patch(color=self.IIb_color, label='IIb')

        IIbMask, IbMask, IcMask, IcBLMask = self.getSNeTypeMasks()

        x = self.pcaCoeffMatrix[:,pcax-1]
        y = self.pcaCoeffMatrix[:,pcay-1]

        #centroids
        nameMask = self.getSNeNameMask(excludeSNe)
        print(np.array(list(self.snidset.keys()))[nameMask])
        print('IIb')
        IIbxmean = np.mean(x[np.logical_and(IIbMask, nameMask)])
        IIbymean = np.mean(y[np.logical_and(IIbMask, nameMask)])
        print('IIb - x: ',x[np.logical_and(IIbMask, nameMask)])
        print('IIb - y: ',y[np.logical_and(IIbMask, nameMask)])
        print('mean = ',(IIbxmean, IIbymean))

        print('Ib')
        Ibxmean = np.mean(x[np.logical_and(IbMask, nameMask)])
        print(x[np.logical_and(IbMask, nameMask)].shape)
        print(x[IbMask].shape)
        Ibymean = np.mean(y[np.logical_and(IbMask, nameMask)])
        print('Ib - x: ',x[np.logical_and(IbMask, nameMask)])
        print('Ib - y: ',y[np.logical_and(IbMask, nameMask)])
        print('mean = ',(Ibxmean, Ibymean))

        print('Ic')
        Icxmean = np.mean(x[np.logical_and(IcMask, nameMask)])
        Icymean = np.mean(y[np.logical_and(IcMask, nameMask)])
        print('Ic - x: ',x[np.logical_and(IcMask, nameMask)])
        print('Ic - y: ',y[np.logical_and(IcMask, nameMask)])
        print("mask mean: ",Ibxmean, Ibymean)
        print("no mask mean: ", np.mean(x[IbMask]), np.mean(y[IbMask]))
        print('mean = ',(Icxmean, Icymean))

        print('IcBL')
        print('IcBL - x: ',x[np.logical_and(IcBLMask, nameMask)])
        print('IcBL - y: ',y[np.logical_and(IcBLMask, nameMask)])
        IcBLxmean = np.mean(x[np.logical_and(IcBLMask, nameMask)])
        IcBLymean = np.mean(y[np.logical_and(IcBLMask, nameMask)])
        print('mean = ',(IcBLxmean, IcBLymean))


        if svm:
            truth = 1*IIbMask + 2*IbMask + 3*IcMask + 4*IcBLMask
            dat = np.column_stack((x,y))
            linsvm = LinearSVC()

            ncv_scores=[]
            for i in range(ncv):
                trainX, testX, trainY, testY = train_test_split(dat, truth, test_size=0.3)
                linsvm.fit(trainX, trainY)
                score = linsvm.score(testX, testY)
                ncv_scores.append(score)
            
                mesh_x, mesh_y = make_meshgrid(x, y, h=0.02)


                Z = linsvm.predict(np.c_[mesh_x.ravel(), mesh_y.ravel()])

                IIbZ = len(Z[Z==1])
                IbZ = len(Z[Z==2])
                IcZ = len(Z[Z==3])
                IcBLZ = len(Z[Z==4])
                colors = []
                if IIbZ != 0:
                    colors.append(self.IIb_color)
                if IbZ != 0:
                    colors.append(self.Ib_color)
                if IcZ != 0:
                    colors.append(self.Ic_color)
                if IcBLZ != 0:
                    colors.append(self.IcBL_color)
                nbins = len(colors)
                cmap_name = 'mymap'
                cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=nbins)
                Z = Z.reshape(mesh_x.shape)
                out = ax.contourf(mesh_x, mesh_y, Z, cmap=cm, alpha=0.2/alphasvm)
                svmsc.append(score)
        if purity:
            nameMask = self.getSNeNameMask(excludeSNe)
            print('rad namemask: ',nameMask)
            IIb_rad_x = np.std(x[np.logical_and(IIbMask, nameMask)]) * std_rad
            Ib_rad_x = np.std(x[np.logical_and(IbMask, nameMask)]) * std_rad
            Ic_rad_x = np.std(x[np.logical_and(IcMask, nameMask)]) * std_rad
            IcBL_rad_x = np.std(x[np.logical_and(IcBLMask, nameMask)]) * std_rad

            IIb_rad_y = np.std(y[np.logical_and(IIbMask, nameMask)]) * std_rad
            Ib_rad_y = np.std(y[np.logical_and(IbMask, nameMask)]) * std_rad
            Ic_rad_y = np.std(y[np.logical_and(IcMask, nameMask)]) * std_rad
            IcBL_rad_y = np.std(y[np.logical_and(IcBLMask, nameMask)]) * std_rad



            print('IIb radx, rady',(IIb_rad_x, IIb_rad_y))
            print('Ib radx, rady',(Ib_rad_x, Ib_rad_y))
            print('Ic radx, rady',(Ic_rad_x, Ic_rad_y))
            print('IcBL radx, rady',(IcBL_rad_x, IcBL_rad_y))


            print("rad mask: ",Ib_rad_x, Ib_rad_y)
            print("rad all: ",np.std(x[IbMask]), np.std(y[IbMask]))
            print('names all: ',np.array(list(self.snidset.keys()))[IbMask])

            names = np.array(list(self.snidset.keys()))
            IIbnames = names[IIbMask]
            dist_x = np.power(x[IIbMask] - IIbxmean, 2)/np.power(2*IIb_rad_x, 2)
            dist_y = np.power(y[IIbMask] - IIbymean, 2)/np.power(2*IIb_rad_y, 2)
            outliers_mask = dist_x + dist_y >= 1
            print('IIb 2std outliers: ', IIbnames[outliers_mask])
            IIb_out_only_mask = np.logical_and(np.logical_not(self.getSNeNameMask(IIbnames[outliers_mask])), IIbMask)
            IIb_no_out_mask = np.logical_and(self.getSNeNameMask(IIbnames[outliers_mask]), IIbMask)

            Ibnames = names[IbMask]
            dist_x = np.power(x[IbMask] - Ibxmean, 2)/np.power(2*Ib_rad_x, 2)
            dist_y = np.power(y[IbMask] - Ibymean, 2)/np.power(2*Ib_rad_y, 2)
            outliers_mask = dist_x + dist_y >= 1
            print('Ib 2std outliers: ', Ibnames[outliers_mask])
            Ib_out_only_mask = np.logical_and(np.logical_not(self.getSNeNameMask(Ibnames[outliers_mask])), IbMask)
            Ib_no_out_mask = np.logical_and(self.getSNeNameMask(Ibnames[outliers_mask]), IbMask)
           
            Icnames = names[IcMask]
            dist_x = np.power(x[IcMask] - Icxmean, 2)/np.power(2*Ic_rad_x, 2)
            dist_y = np.power(y[IcMask] - Icymean, 2)/np.power(2*Ic_rad_y, 2)
            outliers_mask = dist_x + dist_y >= 1
            print('Ic 2std outliers: ', Icnames[outliers_mask])
            Ic_out_only_mask = np.logical_and(np.logical_not(self.getSNeNameMask(Icnames[outliers_mask])), IcMask)
            Ic_no_out_mask = np.logical_and(self.getSNeNameMask(Icnames[outliers_mask]), IcMask)

            IcBLnames = names[IcBLMask]
            dist_x = np.power(x[IcBLMask] - IcBLxmean, 2)/np.power(2*IcBL_rad_x, 2)
            dist_y = np.power(y[IcBLMask] - IcBLymean, 2)/np.power(2*IcBL_rad_y, 2)
            outliers_mask = dist_x + dist_y >= 1
            print('IcBL 2std outliers: ', IcBLnames[outliers_mask])
            IcBL_out_only_mask = np.logical_and(np.logical_not(self.getSNeNameMask(IcBLnames[outliers_mask])), IcBLMask)
            IcBL_no_out_mask = np.logical_and(self.getSNeNameMask(IcBLnames[outliers_mask]), IcBLMask)


            ellipse_IIb = mpatches.Ellipse((IIbxmean, IIbymean),2*IIb_rad_x,2*IIb_rad_y, \
                                            color=self.IIb_ellipse_color, alpha=0.1/alphaell, fill=False, edgecolor=self.IIb_color, linewidth=4.0)
            ellipse_Ib = mpatches.Ellipse((Ibxmean, Ibymean),2*Ib_rad_x,2*Ib_rad_y, color=self.Ib_color,\
                                           alpha=0.1/alphaell, fill=False, edgecolor=self.Ib_color, linewidth=4.0)
            ellipse_Ic = mpatches.Ellipse((Icxmean, Icymean),2*Ic_rad_x,2*Ic_rad_y, color=self.Ic_color,\
                                           alpha=0.1/alphaell, fill=False, edgecolor=self.Ic_color, linewidth=4.0)
            ellipse_IcBL = mpatches.Ellipse((IcBLxmean, IcBLymean),2*IcBL_rad_x,2*IcBL_rad_y, color=self.IcBL_ellipse_color,\
                                             alpha=0.1/alphaell,fill=False, edgecolor=self.IcBL_color, linewidth=4.0)
            ax.add_patch(ellipse_IIb)
            ax.add_patch(ellipse_Ib)
            ax.add_patch(ellipse_Ic)
            ax.add_patch(ellipse_IcBL)

        if markOutliers:

            ax.scatter(x[IIb_no_out_mask], y[IIb_no_out_mask], color=self.IIb_color, edgecolors='k',s=200,alpha=1,linewidth=2.0)
            ax.scatter(x[Ib_no_out_mask], y[Ib_no_out_mask], color=self.Ib_color, edgecolors='k',s=200,alpha=1, linewidth=2.0)
            ax.scatter(x[Ic_no_out_mask], y[Ic_no_out_mask], color=self.Ic_color, edgecolors='k',s=200,alpha=1, linewidth=2.0)
            ax.scatter(x[IcBL_no_out_mask], y[IcBL_no_out_mask], color=self.IcBL_color, edgecolors='k',s=200,alpha=1, linewidth=2.0)


            ax.scatter(x[IIb_out_only_mask], y[IIb_out_only_mask], color=self.IIb_color, edgecolors='k',s=400,alpha=1,linewidth=2.0, marker='*')
            ax.scatter(x[Ib_out_only_mask], y[Ib_out_only_mask], color=self.Ib_color, edgecolors='k',s=400,alpha=1, linewidth=2.0, marker='*')
            ax.scatter(x[Ic_out_only_mask], y[Ic_out_only_mask], color=self.Ic_color, edgecolors='k',s=400,alpha=1, linewidth=2.0, marker='*')
            ax.scatter(x[IcBL_out_only_mask], y[IcBL_out_only_mask], color=self.IcBL_color, edgecolors='k',s=400,alpha=1, linewidth=2.0, marker='*')

        else:
            ax.scatter(x[IIbMask], y[IIbMask], color=self.IIb_color, edgecolors='k',s=200,alpha=1,linewidth=2.0)
            ax.scatter(x[IbMask], y[IbMask], color=self.Ib_color, edgecolors='k',s=200,alpha=1, linewidth=2.0)
            ax.scatter(x[IcMask], y[IcMask], color=self.Ic_color, edgecolors='k',s=200,alpha=1, linewidth=2.0)
            ax.scatter(x[IcBLMask], y[IcBLMask], color=self.IcBL_color, edgecolors='k',s=200,alpha=1, linewidth=2.0)

        ax.set_xlim((np.min(x)-.2,np.max(x)+.2))
        ax.set_ylim((np.min(y)-.2,np.max(y)+.2))

        ax.set_ylabel('PCA Comp %d'%(pcay),fontsize=20)
        ax.set_xlabel('PCA Comp %d'%(pcax), fontsize=20)
        if svm:
            avgsc = np.mean(np.array(ncv_scores))
            stdsc = np.std(np.array(ncv_scores))
            ax.legend(handles=[Ic_patch, IcBL_patch, IIb_patch, Ib_patch],\
                            title='SVM Test Score = %.2f'%(avgsc), loc='upper right', ncol=1,fancybox=True, prop={'size':30},fontsize=30)
        else:
            ax.legend(handles=[Ic_patch, IcBL_patch, IIb_patch, Ib_patch], fontsize=18)
        ax.minorticks_on()
        ax.tick_params(
                    axis='both',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    labelsize=20) # labels along the bottom edge are off
        if svm:
            return f, svmsc, avgsc, stdsc
        return f, svmsc



    def purityEllipse(self, std_rad, ncomp_array):
        """
        Calculates the purity of the SESN regions

        Returns
        -------

        """
        ncomp_array = np.array(ncomp_array) - 1
        IIbMask, IbMask, IcMask, IcBLMask = self.getSNeTypeMasks()
        maskDict = {'IIb':IIbMask, 'Ib':IbMask, 'IcBL':IcBLMask, 'Ic':IcMask}
        keys = ['IIb', 'Ib', 'IcBL', 'Ic']
        masks = [IIbMask, IbMask, IcBLMask, IcMask]
        purity_rad_arr = []
        for key,msk in zip(keys,masks):
            centroid = np.mean(self.pcaCoeffMatrix[:,ncomp_array][msk], axis=0)
            std = np.std(self.pcaCoeffMatrix[:,ncomp_array][msk], axis=0)
            print('centroid', centroid)
            dist_from_centroid = np.abs(self.pcaCoeffMatrix[:,ncomp_array][msk] - centroid)
            mean_dist_from_centroid = np.mean(dist_from_centroid, axis=0)
            print('mean dist from centroid: ', mean_dist_from_centroid)
            std_dist_all_components = np.std(dist_from_centroid, axis=0)
            print('std dist from centroid: ', std_dist_all_components)
            purity_rad_all = mean_dist_from_centroid + std_rad * std_dist_all_components
            print('purity rad all components: ', purity_rad_all)
            purity_rad_arr.append(std)


            ellipse_cond = np.sum(np.power((self.pcaCoeffMatrix[:,ncomp_array] - centroid), 2)/\
                                  np.power(purity_rad_all, 2), axis=1)
            print('ellipse condition: ', ellipse_cond)
            purity_msk = ellipse_cond < 1

            print(key)
            print('purity radius: ', purity_rad_all)
            print('# of SNe within purity ellipse for type '+key+': ',np.sum(purity_msk))
            names_within_purity_rad = self.pcaNames[purity_msk]
            correct_names = self.pcaNames[msk]
            correct_msk = np.isin(names_within_purity_rad, correct_names)
            print('# of correct SNe '+key+': ', np.sum(correct_msk))
        return keys, purity_rad_arr


    def pcaPlotly(self, pcaxind, pcayind, std_rad, excludeSNe=[]):
        """
        Makes PCA plot interactively using Plotly. Does not show SVM regions.

        Parameters
        ----------
        pcaxind : int
        pcayind : int
        std_rad : int
        excludeSNe : list

        Returns
        -------

        """
        IIbmask, Ibmask, Icmask, IcBLmask = self.getSNeTypeMasks()
        pcax = self.pcaCoeffMatrix[:,pcaxind - 1]
        pcay = self.pcaCoeffMatrix[:,pcayind - 1]
        col_red = 'rgba(152,0,0,1)'
        col_blue = 'rgba(0,152,152,1)'
        col_green = 'rgba(0,152,0,1)'
        col_black = 'rgba(0,0,0,152)'
        col_purp = 'rgba(186,85,211, 0.8)'

        traceIIb=go.Scatter(x=pcax[IIbmask], y=pcay[IIbmask], mode='markers',\
                            marker=dict(size=10, line=dict(width=1), color=col_green, opacity=1), \
                            text=np.array([nm+'_'+ph for nm,ph in zip(self.pcaNames, self.pcaPhases)])[IIbmask], name='IIb')
        
        traceIb=go.Scatter(x=pcax[Ibmask], y=pcay[Ibmask], mode='markers',\
                            marker=dict(size=10, line=dict(width=1), color=col_purp, opacity=1), \
                            text=np.array([nm+'_'+ph for nm,ph in zip(self.pcaNames, self.pcaPhases)])[Ibmask], name='Ib')
        
        traceIc=go.Scatter(x=pcax[Icmask], y=pcay[Icmask], mode='markers',\
                            marker=dict(size=10, line=dict(width=1), color=col_red, opacity=1), \
                            text=np.array([nm+'_'+ph for nm,ph in zip(self.pcaNames, self.pcaPhases)])[Icmask], name='Ic')
        
        traceIcBL=go.Scatter(x=pcax[IcBLmask], y=pcay[IcBLmask], mode='markers',\
                            marker=dict(size=10, line=dict(width=1), color=col_black, opacity=1), \
                            text=np.array([nm+'_'+ph for nm,ph in zip(self.pcaNames, self.pcaPhases)])[IcBLmask], name='IcBL')
        data = [traceIIb, traceIb, traceIc, traceIcBL]



        nameMask = self.getSNeNameMask(excludeSNe)
        print(np.array(list(self.snidset.keys()))[nameMask])
        IIbxmean = np.mean(pcax[np.logical_and(IIbmask, nameMask)])
        IIbymean = np.mean(pcay[np.logical_and(IIbmask, nameMask)])
        Ibxmean = np.mean(pcax[np.logical_and(Ibmask, nameMask)])
        print(pcax[np.logical_and(Ibmask, nameMask)].shape)
        print(pcax[Ibmask].shape)
        Ibymean = np.mean(pcay[np.logical_and(Ibmask, nameMask)])
        Icxmean = np.mean(pcax[np.logical_and(Icmask, nameMask)])
        Icymean = np.mean(pcay[np.logical_and(Icmask, nameMask)])
        print("mask mean: ",Ibxmean, Ibymean)
        print("no mask mean: ", np.mean(pcax[Ibmask]), np.mean(pcay[Ibmask]))
        IcBLxmean = np.mean(pcax[np.logical_and(IcBLmask, nameMask)])
        IcBLymean = np.mean(pcay[np.logical_and(IcBLmask, nameMask)])
        nameMask = self.getSNeNameMask(excludeSNe)
        print('rad namemask: ',nameMask)
        IIbradx = np.std(pcax[np.logical_and(IIbmask, nameMask)]) * std_rad
        Ibradx = np.std(pcax[np.logical_and(Ibmask, nameMask)]) * std_rad
        Icradx = np.std(pcax[np.logical_and(Icmask, nameMask)]) * std_rad
        IcBLradx = np.std(pcax[np.logical_and(IcBLmask, nameMask)]) * std_rad
 
        IIbrady = np.std(pcay[np.logical_and(IIbmask, nameMask)]) * std_rad
        Ibrady = np.std(pcay[np.logical_and(Ibmask, nameMask)]) * std_rad
        Icrady = np.std(pcay[np.logical_and(Icmask, nameMask)]) * std_rad
        IcBLrady = np.std(pcay[np.logical_and(IcBLmask, nameMask)]) * std_rad

        layout = go.Layout(autosize=False,
               width=1000,
               height=700,
               annotations=[
                   dict(
                       x=1.05,
                       y=1.025,
                       showarrow=False,
                       text='Phases: [%.2f, %.2f]'%(self.phasemin, self.phasemax),
                       xref='paper',
                       yref='paper'
                   )],
               xaxis=dict(
                   title='PCA%i'%(pcaxind),
                   titlefont=dict(
                       family='Courier New, monospace',
                       size=30,
                       color='black'
                   ),
               ),
               yaxis=dict(
                   title='PCA%i'%(pcayind),
                   titlefont=dict(
                       family='Courier New, monospace',
                       size=30,
                       color='black'
                   ),
               ), shapes=[
                   {
                       'type': 'circle',
                       'xref': 'x',
                       'yref': 'y',
                       'x0': IIbxmean-IIbradx,
                       'y0': IIbymean - IIbrady,
                       'x1': IIbxmean+IIbradx,
                       'y1': IIbymean + IIbrady,
                       'opacity': 0.2,
                       'fillcolor': col_green,
                       'line': {
                           'color': col_green,
                       },
                   },
               {
                       'type': 'circle',
                       'xref': 'x',
                       'yref': 'y',
                       'x0': Ibxmean - Ibradx,
                       'y0': Ibymean - Ibrady,
                       'x1': Ibxmean + Ibradx,
                       'y1': Ibymean + Ibrady,
                       'opacity': 0.2,
                       'fillcolor': col_purp,
                       'line': {
                           'color': col_purp,
                       },
                   },{
                       'type': 'circle',
                       'xref': 'x',
                       'yref': 'y',
                       'x0': Icxmean - Icradx,
                       'y0': Icymean - Icrady,
                       'x1': Icxmean + Icradx,
                       'y1': Icymean + Icrady,
                       'opacity': 0.2,
                       'fillcolor': col_red,
                       'line': {
                           'color': col_red
                       }
                   },{
                       'type': 'circle',
                       'xref': 'x',
                       'yref': 'y',
                       'x0': IcBLxmean - IcBLradx,
                       'y0': IcBLymean - IcBLrady,
                       'x1': IcBLxmean + IcBLradx,
                       'y1': IcBLymean + IcBLrady,
                       'opacity': 0.2,
                       'fillcolor': col_black,
                       'line': {
                           'color': col_black
                       }
                   }]
           )
        fig = go.Figure(data=data, layout=layout)
        return fig

    def cornerplotPCA(self, ncomp, figsize, svm=False, ncv=1):
        """
        Plots the 2D marginalizations of the PCA decomposition in a corner plot.

        Parameters
        ----------
        ncomp : int
            Number of PCA components in corner plot
        figsize : tuple
        svm : Boolean
            Calculates SVM scores if True
        ncv : int
            Number of cross validation runs

        Returns
        -------
        f : plt.figure
        svm_highscore : float
            highest svm avg score
        svm_x : int
            x ind of best pca component
        svm_y : int
            y ind of best pca component
        means_table : pandas Data Table
            average svm scores of the 2D marginalizations
        std_table : pandas Data Table
            standard deviations of svm scores for
            the 2D marginalizations

        """
        red_patch = mpatches.Patch(color=self.Ic_color, label='Ic')
        cyan_patch = mpatches.Patch(color=self.Ib_color, label='Ib')
        black_patch = mpatches.Patch(color=self.IcBL_color, label='IcBL Smoothed')
        green_patch = mpatches.Patch(color=self.IIb_color, label='IIb')

        IIbMask, IbMask, IcMask, IcBLMask = self.getSNeTypeMasks()
        svm_highscore = 0.0
        svm_x = -1
        svm_y = -1

        means_table = np.zeros((ncomp,ncomp))
        std_table = np.zeros((ncomp,ncomp))
        f = plt.figure(figsize=figsize)
        for i in range(ncomp):
            for j in range(ncomp):
                if i > j:
                    plotNumber = ncomp * i + j + 1
                    print(plotNumber)
                    plt.subplot(ncomp, ncomp, plotNumber)
                    y = self.pcaCoeffMatrix[:,i]
                    x = self.pcaCoeffMatrix[:,j]

                    #centroids
                    IIbxmean = np.mean(x[IIbMask])
                    IIbymean = np.mean(y[IIbMask])
                    Ibxmean = np.mean(x[IbMask])
                    Ibymean = np.mean(y[IbMask])
                    Icxmean = np.mean(x[IcMask])
                    Icymean = np.mean(y[IcMask])
                    IcBLxmean = np.mean(x[IcBLMask])
                    IcBLymean = np.mean(y[IcBLMask])
                    plt.scatter(IIbxmean, IIbymean, color=self.IIb_color, alpha=0.5, s=100)
                    plt.scatter(Ibxmean, Ibymean, color=self.Ib_color, alpha=0.5, s=100)
                    plt.scatter(Icxmean, Icymean, color=self.Ic_color, alpha=0.5, s=100)
                    plt.scatter(IcBLxmean, IcBLymean, color=self.IcBL_color, alpha=0.5, s=100)

                    if svm:
                        truth = 1*IIbMask + 2*IbMask + 3*IcMask + 4*IcBLMask
                        dat = np.column_stack((x,y))

                        ncv_scores=[]
                        for cvit in range(ncv):
                            trainX, testX, trainY, testY = train_test_split(dat, truth, test_size=0.3)
                            linsvm = LinearSVC()
                            linsvm.fit(trainX, trainY)
                            score = linsvm.score(testX, testY)
                            ncv_scores.append(score)
                        score = np.mean(ncv_scores)
                        std = np.std(ncv_scores)
                        means_table[j,i] = score
                        means_table[i,j] = score
                        std_table[j,i] = std
                        std_table[i,j] = std

                        if score > svm_highscore:
                            svm_highscore = score
                            svm_x = j+1
                            svm_y = i+1

                    plt.scatter(x[IIbMask], y[IIbMask], color=self.IIb_color, alpha=1)
                    plt.scatter(x[IbMask], y[IbMask], color=self.Ib_color, alpha=1)
                    plt.scatter(x[IcMask], y[IcMask], color=self.Ic_color, alpha=1)
                    plt.scatter(x[IcBLMask], y[IcBLMask], color=self.IcBL_color, alpha=1)

                    plt.xlim((np.min(self.pcaCoeffMatrix[:,j])-2,np.max(self.pcaCoeffMatrix[:,j])+2))
                    plt.ylim((np.min(self.pcaCoeffMatrix[:,i])-2,np.max(self.pcaCoeffMatrix[:,i])+2))

                    if j == 0:
                        plt.ylabel('PCA Comp %d'%(i+1))
                    if i == ncomp - 1:
                        plt.xlabel('PCA Comp %d'%(j+1))
        plt.subplot(5,5,9)#########################################################
        plt.axis('off')
        plt.legend(handles=[red_patch, cyan_patch, black_patch, green_patch])
        if svm:
            return f, svm_highscore, svm_x, svm_y, means_table, std_table
        return f
