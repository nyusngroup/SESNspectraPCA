# Class written to run PCA analysis on SNID supernova spectra.
# Author: Marc Williamson
# Date created: 3/01/2018

import numpy as np
import matplotlib.pyplot as plt
import sys, os, glob, re, copy
import sklearn
from sklearn.decomposition import PCA
import plotly.plotly as ply
import plotly.graph_objs as go
import plotly.tools as tls
import pickle
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

try:
    import pidly
except ImportError:
    print 'You do not have pidly installed! Install it with pip, or from https://github.com/anthonyjsmith/pIDLy'
    print 'In order to use smoothed IcBL spectra, you need pidly and SNspecFFTsmooth.pro from https://github.com/nyusngroup/SESNspectraLib'
    print 'You can also use the example pickled object provided that has had IcBL smoothed already.'

# Global Variables
firstSNIDWavelength = '2501.69' # the first observed wavelength in all the SNID files


# Private helper functions



# The following function finds the observed phase closest to the loadPhase value
# specified by the user for each SNe. It returns phaseCols, a list of the column
# number in the SNID file of the desired phase (NEED to add 1 because there is a 
# Wavelength column in the SNID file). It also returns phaseArr, an array of the 
# closest phase for each SNe in the sample.
def findClosestObsPhase(phases, loadPhase):        
    phaseCols = []
    phaseArr = []
    for i in range(len(phases)):
        idx = (np.abs(phases[i] - loadPhase)).argmin()
        phaseCols.append(idx)
        phaseArr.append(phases[i][idx])
    phaseArr = np.array(phaseArr)
    phaseCols = np.array(phaseCols)
    return phaseCols, phaseArr



# Function to get masks of IIb, Ib, Ic, and IcBL SNe. These masks are used internally
# for plotting the different SNe types in different colors.
def getSNeTypeMasks(sneTypes):
    IIbMask = np.array([np.array_equal(typeArr,[2,4]) for typeArr in sneTypes])
    
    tmp1 = np.array([np.array_equal(typeArr,[2,1]) for typeArr in sneTypes])
    tmp2 = np.logical_or(tmp1, np.array([np.array_equal(typeArr,[2,2]) for typeArr in sneTypes]))
    tmp3 = np.logical_or(tmp2, np.array([np.array_equal(typeArr,[2,3]) for typeArr in sneTypes]))
    IbMask = np.logical_or(tmp3, np.array([np.array_equal(typeArr,[2,5]) for typeArr in sneTypes]))
    
    IcBLMask = np.array([np.array_equal(typeArr,[3,4]) for typeArr in sneTypes])

    tmp1 = np.array([np.array_equal(typeArr,[3,1]) for typeArr in sneTypes])
    tmp2 = np.logical_or(tmp1, np.array([np.array_equal(typeArr,[3,2]) for typeArr in sneTypes]))
    IcMask = np.logical_or(tmp2, np.array([np.array_equal(typeArr,[3,3]) for typeArr in sneTypes]))

    return IIbMask, IbMask, IcMask, IcBLMask




class NovaPCA:

# Initialize the NovaPCA object with a path to the directory containing all 
# the SNID spectra.
    def __init__(self, snidDirPath):
        self.spectraMatrix = None
        self.pcaCoeffMatrix = None
        self.obsSNIDPhases = None
        self.sneNames = None
        self.sneTypes = None
        self.skiprows = None
        self.phaseCols = None
        self.spectraMean = None
        self.spectraStd = None
        
        self.maskAttributes = ['spectraMatrix','pcaCoeffMatrix','obsSNIDPhases','sneNames','sneTypes',\
                              'skiprows', 'phaseCols', 'spectraMean', 'spectraStd']

        self.sklearnPCA = None
        self.evecs = None
        self.evals = None
        self.evals_cs = None

        self.wavelengths = None
        self.loadPhase = None
        self.phaseWidth = None
        self.maskList = []
        self.snidDirPath = snidDirPath
        
        return

# Method to load SNID spectra of the types specified in loadTypes. 
# The SNID type structure is listed at the end of this file.
# Arguments:
#     loadTypes -- list of 2tuples specifying which SNe types to load.
#                  See SNID structure at end of file.
#                  Ex) >>> loadTypes = [(1,2), (2,2), (3,2)] loads Ia-norm, Ib-norm, and Ic-norm.
#     phaseType -- int, either 0 for phases measured relative to max light, or 1 
#                  for phases measured relative to first observation.
#     loadPhase -- float, what phase you want to load the nearest observed spectra of for each SNe.
#     loadPhaseRangeWidth -- float, width of phase range you want to allow, ie phase = 15 +/- 5 days.
#     minwvl/maxwvl -- wavelength cutoffs for loading the spectra.

    def loadSNID(self, loadTypes, phaseType, loadPhase, loadPhaseRangeWidth, minwvl, maxwvl):

        self.loadPhase = loadPhase
        self.phaseWidth = loadPhaseRangeWidth
        self.minwvl = minwvl
        self.maxwvl = maxwvl

        temp = os.getcwd()
        os.chdir(self.snidDirPath)
        allSpec = glob.glob('*.lnw')
        os.chdir(temp)

        snePaths = []
        sneNames = []
        sneTypeList = []
        skiprows = []
        phaseTypeList = []
        phases = []

        for specName in allSpec:
            path = self.snidDirPath + specName
            with open(path, 'r') as f:
                lines = f.readlines()
                header = lines[0].split()
                snType = int(header[-2])
                snSubtype = int(header[-1])
                typeTup = (snType, snSubtype)
                if typeTup in loadTypes:
                    snePaths.append(path)
                    sneNames.append(specName[:-4])
                    sneTypeList.append(typeTup)
                    for i in range(len(lines) - 1):
                        line = lines[i]
                        if firstSNIDWavelength in line:
                            skiprows.append(i)
                            phaseRow = lines[i - 1].split()
                            phaseTypeList.append(int(phaseRow[0]))
                            phases.append(np.array([float(ph) for ph in phaseRow[1:]]))
                            break
        phaseTypeList = np.array(phaseTypeList)
        sneTypeList = np.array(sneTypeList)
        sneNames = np.array(sneNames)
        snePaths = np.array(snePaths)
        skiprows = np.array(skiprows)
        print len(phases) #################
        phaseCols, phaseArr = findClosestObsPhase(phases, loadPhase)
        
        spectra = []
        for i in range(len(snePaths)):
            spec = snePaths[i]
            skiprow = skiprows[i]
            s = np.loadtxt(spec, skiprows=skiprow, usecols=(0,phaseCols[i] + 1)) #Note the +1 becase in the SNID files there is a column (0) for wvl
            mask = np.logical_and(s[:,0] > self.minwvl, s[:,0] < self.maxwvl)
            s = s[mask]
            spectra.append(s)

        wavelengths = np.array(spectra[0][:,0])
        specMat = np.ndarray((len(spectra), spectra[0].shape[0]))
        for i, spec in enumerate(spectra):
            specMat[i,:] = spec[:,1]

        phaseRangeMask = np.logical_and(phaseArr > (loadPhase - loadPhaseRangeWidth), phaseArr < (loadPhase + loadPhaseRangeWidth))
        phaseMask = np.logical_and(phaseTypeList == 0, phaseRangeMask)


        specMat = specMat[phaseMask]
        sneNames = sneNames[phaseMask]
        phaseArr = phaseArr[phaseMask]
        sneTypeList = sneTypeList[phaseMask]
        skiprows = skiprows[phaseMask]
        phaseCols = phaseCols[phaseMask]


        self.sneNames = np.copy(sneNames)
        self.obsSNIDPhases = np.copy(phaseArr)
        self.spectraMatrix = np.copy(specMat)
        self.sneTypes = np.copy(sneTypeList)
        self.wavelengths = np.copy(wavelengths)
        self.skiprows = np.copy(skiprows)
        self.phaseCols = np.copy(phaseCols)

        return

# This Method smooths the broadline Ic spectra and outputs
# a plot of the smoothed vs unsmoothed spectra for verification.

    def smoothIcBL(self):

        IcBLMask = np.array([np.array_equal(arr, np.array([3,4])) for arr in self.sneTypes])
        idl = pidly.IDL()
        idl('.COMPILE type default lmfit linear_fit powerlaw_fit integ binspec SNspecFFTsmooth')
        IcBLSmoothedMatrix = []
        IcBLPreSmooth = []
        IcBLPreSmoothIDL = []
        
        for i in range(len(IcBLMask[IcBLMask == True])):
            specName = self.sneNames[IcBLMask][i]
            print specName
            specPath = self.snidDirPath + specName +'.lnw'
            skiprow = self.skiprows[IcBLMask][i]
            usecol = self.phaseCols[IcBLMask][i] + 1 # add 1 because SNID file has a wavelength column
            s = np.loadtxt(specPath, skiprows=skiprow, usecols=(0,usecol))
            if i==0:
                IcBLWvl = s[:,0]
            with open('tmp_spec.txt', 'w') as f:
                for j in range(s.shape[0]):
                    f.write('        %.4f        %.7f\n'%(s[j,0],s[j,1]+10.0))
                f.close()
            idl('readcol, "tmp_spec.txt", w, f')
            idl('SNspecFFTsmooth, w, f, 3000, f_ft, f_std, sep_vel')
            IcBLPreSmooth.append(s[:,1])
            IcBLSmoothedMatrix.append(idl.f_ft)
            IcBLPreSmoothIDL.append(idl.f)
            
        IcBLPreSmooth = np.array(IcBLPreSmooth)
        IcBLSmoothedMatrix = np.array(IcBLSmoothedMatrix)
        IcBLPreSmoothIDL = np.array(IcBLPreSmoothIDL)

        smoothMean = np.nanmean(IcBLSmoothedMatrix, axis=1)
        smoothStd = np.nanstd(IcBLSmoothedMatrix, axis=1)

        originalSmoothSpectra = np.copy(IcBLSmoothedMatrix)
        IcBLSmoothedMatrixT = (IcBLSmoothedMatrix.T - smoothMean) / smoothStd
        IcBLSmoothedMatrix = IcBLSmoothedMatrixT.T

        preSmoothMean = np.nanmean(IcBLPreSmooth, axis=1)
        preSmoothStd = np.nanstd(IcBLPreSmooth, axis=1)
        originalPreSmoothSpectra = np.copy(IcBLPreSmooth)
        IcBLPreSmoothT = (IcBLPreSmooth.T - preSmoothMean) / preSmoothStd
        IcBLPreSmooth = IcBLPreSmoothT.T

        f = plt.figure(figsize=(15,50))
        nshow = IcBLSmoothedMatrix.shape[0]
        for i in range(nshow):
            plt.subplot(nshow, 1, i + 1)
            name = self.sneNames[IcBLMask][i]
            plt.plot(IcBLWvl, IcBLSmoothedMatrix[i], label='smoothed '+name, color='k')
            plt.plot(IcBLWvl, IcBLPreSmooth[i], label='pre smoothed '+name, color='r')
            if i == 0:
                plt.title('Smoothed IcBL spectra %d$\pm$%d'%(self.loadPhase, self.phaseWidth))
            if i == nshow - 1:
                plt.xlabel('Wavelength (Angstroms)')
            plt.ylabel('Rel Flux')
            plt.legend(fontsize=12)
        for i, sn in enumerate(self.sneNames[IcBLMask]):
            print sn
            smoothSpec = IcBLSmoothedMatrix[i]
            mask = np.logical_and(IcBLWvl > self.minwvl, IcBLWvl < self.maxwvl)
            smoothSpec = smoothSpec[mask]
            ind = np.where(self.sneNames == sn)[0][0]
            self.spectraMatrix[ind] = smoothSpec



        return f

# Preprocessing replaces 0.0 values with NaN. It also removes the mean of each spectrum
# and scales each spectrum to have unitary std.
    def preprocess(self):
        for i in range(self.spectraMatrix.shape[0]):
            self.spectraMatrix[i][self.spectraMatrix[i] == 0] = np.nan
        spectraMean = np.nanmean(self.spectraMatrix, axis=1)
        spectraStd = np.nanstd(self.spectraMatrix, axis=1)
        spectraMatrixT = (self.spectraMatrix.T - spectraMean)/spectraStd
        self.spectraMatrix = spectraMatrixT.T
        self.spectraMean = spectraMean
        self.spectraStd = spectraStd
        return



    def wavelengthRebin(self, smoothing):
        nrows, ncols = self.spectraMatrix.shape
        tmp = np.reshape(self.spectraMatrix, (nrows, ncols/smoothing, smoothing))
        self.spectraMatrix = np.nanmean(tmp, axis=2)
        wvrows = self.wavelengths.shape[0]
        wvcols = 1
        wvtemp = np.reshape(self.wavelengths, (wvrows/smoothing, 1, smoothing))
        self.wavelengths = np.nanmean(wvtemp, axis=2)
        return

# This method takes a user specified mask, and applies it to all the maskable 
# attributes of a NovaPCA instance. If the user sets savecopy=True, then this 
# method first copies the original NovaPCA instance before applying the mask
# and returns the old instance to the user.
    def applyMask(self, mask, savecopy=False):
        if savecopy:
            preMask = copy.deepcopy(self)
        for attr in self.maskAttributes:
            attrObj = getattr(self, attr)
            if not attrObj is None:
                setattr(self, attr, attrObj[mask])
        if savecopy: 
            return preMask
        return

# The save method pickles the NovaPCA object.
    def save(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self, f)
        f.close()
        return

# The load method loads a saved pickle file.
    def load(self, filename):
        f = open(filename, 'rb')
        loadSelf = pickle.load(f)
        f.close()
        return loadSelf

# Calculate PCA decomposition
    def calculatePCA(self):
        pca = PCA()
        pca.fit(self.spectraMatrix)
        self.sklearnPCA = pca
        self.evecs = pca.components_
        self.evals = pca.explained_variance_ratio_
        self.evals_cs = self.evals.cumsum()
        self.pcaCoeffMatrix = np.dot(self.evecs, self.spectraMatrix.T).T
        return

# Plot TSNE embedding

    def plotTSNE(self, nPCAComponents):
        f = plt.figure()
        model = TSNE(n_components=2, random_state=0)
        tsneSpec = model.fit_transform(self.pcaCoeffMatrix[:,0:nPCAComponents])

        IIbMask, IbMask, IcMask, IcBLMask = getSNeTypeMasks(self.sneTypes)
        plt.scatter(tsneSpec[:,0][IIbMask], tsneSpec[:,1][IIbMask], color='g')
        plt.scatter(tsneSpec[:,0][IbMask], tsneSpec[:,1][IbMask], color='c')
        plt.scatter(tsneSpec[:,0][IcMask], tsneSpec[:,1][IcMask], color='r')
        plt.scatter(tsneSpec[:,0][IcBLMask], tsneSpec[:,1][IcBLMask], color='k')
        plt.title('TSNE Projection from PCA')
        plt.xlabel('TSNE Component 0')
        plt.ylabel('TSNE Component 1')
        return f


# Plot 2D Corner plot of PCA components

    def cornerplotPCA(self, ncomp, figsize):
        red_patch = mpatches.Patch(color='red', label='Ic')
        cyan_patch = mpatches.Patch(color='cyan', label='Ib')
        black_patch = mpatches.Patch(color='black', label='IcBL Smoothed')
        green_patch = mpatches.Patch(color='green', label='IIb')

        IIbMask, IbMask, IcMask, IcBLMask = getSNeTypeMasks(self.sneTypes)

        f = plt.figure(figsize=figsize)
        for i in range(ncomp):
            for j in range(ncomp):
                if i > j:
                    plotNumber = ncomp * i + j + 1
                    plt.subplot(ncomp, ncomp, plotNumber)
                    x = self.pcaCoeffMatrix[:,i]
                    y = self.pcaCoeffMatrix[:,j]

                    #centroids
                    IIbxmean = np.mean(x[IIbMask])
                    IIbymean = np.mean(y[IIbMask])
                    Ibxmean = np.mean(x[IbMask])
                    Ibymean = np.mean(y[IbMask])
                    Icxmean = np.mean(x[IcMask])
                    Icymean = np.mean(y[IcMask])
                    IcBLxmean = np.mean(x[IcBLMask])
                    IcBLymean = np.mean(y[IcBLMask])
                    plt.scatter(IIbymean, IIbxmean, color='g', alpha=0.5, s=100)
                    plt.scatter(Ibymean, Ibxmean, color='c', alpha=0.5, s=100)
                    plt.scatter(Icymean, Icxmean, color='r', alpha=0.5, s=100)
                    plt.scatter(IcBLymean, IcBLxmean, color='k', alpha=0.5, s=100)

                    plt.scatter(y[IIbMask], x[IIbMask], color='g', alpha=1)
                    plt.scatter(y[IbMask], x[IbMask], color='c', alpha=1)
                    plt.scatter(y[IcMask], x[IcMask], color='r', alpha=1)
                    plt.scatter(y[IcBLMask], x[IcBLMask], color='k', alpha=1)

                    plt.xlim((np.min(self.pcaCoeffMatrix[:,j])-2,np.max(self.pcaCoeffMatrix[:,j])+2))
                    plt.ylim((np.min(self.pcaCoeffMatrix[:,i])-2,np.max(self.pcaCoeffMatrix[:,i])+2))

                    if j == 0:
                        plt.ylabel('PCA Comp %d'%(i))
                    if i == ncomp - 1:
                        plt.xlabel('PCA Comp %d'%(j))
        plt.subplot(5,5,9)#########################################################
        plt.axis('off')
        plt.legend(handles=[red_patch, cyan_patch, black_patch, green_patch])
        plt.text(-3.0,1.3,'Smoothed IcBL PCA Component 2D Marginalizations (Phase %d$\pm$%d Days)'%(self.loadPhase, self.phaseWidth),fontsize=16)
        return f








# Plot reconstructed spectra

    def reconstructSpectra(self, nrecon, nPCAComponents):
        randomSpec = np.random.randint(0,self.spectraMatrix.shape[0], nrecon)
        #randomSpec = np.where(self.sneNames == 'sn2004gt')[0]

        self.sampleMean = np.nanmean(self.spectraMatrix, axis=0)
        
        for j, spec in enumerate(randomSpec):
            specName = self.sneNames[spec]
            trueSpec = self.spectraMatrix[spec]
            pcaCoeff = np.dot(self.evecs, (trueSpec - self.sampleMean))
            f = plt.figure(j, figsize=(15,20))
            plt.tick_params(axis='both', which='both', bottom='off', top='off',\
                            labelbottom='off', right='off', left='off', labelleft='off')
            plt.title(specName +' PCA Reconstruction',fontsize=16)
            f.subplots_adjust(hspace=0, top=0.95, bottom=0.1, left=0.12, right=0.93)

            for i, n in enumerate(nPCAComponents):
                ax = f.add_subplot(411 + i)
                ax.plot(self.wavelengths, trueSpec, '-', c='gray')
                ax.plot(self.wavelengths, self.sampleMean + (np.dot(pcaCoeff[:n], self.evecs[:n])), '-k')
                if i < len(nPCAComponents) - 1:
                    plt.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='off',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    labelbottom='off') # labels along the bottom edge are off
                ax.set_ylim(-5,5)
                ax.set_ylabel('flux', fontsize=16)

                if n == 0:
                    text = 'mean'
                elif n == 1:
                    text = "1 component\n"
                    text += r"$(\sigma^2_{tot} = %.2f)$" % self.evals_cs[n - 1]
                else:
                    text = "%i components\n" % n
                    text += r"$(\sigma^2_{tot} = %.2f)$" % self.evals_cs[n - 1]
                ax.text(0.02, 0.93, text, fontsize=20,ha='left', va='top', transform=ax.transAxes)
                f.axes[-1].set_xlabel(r'${\rm wavelength\ (\AA)}$',fontsize=16)
        return f







# Plot eigenspectra

    def plotEigenspectra(self, figsize, nshow):
        f = plt.figure(figsize=figsize)
        for i, ev in enumerate(self.evecs[:nshow]):
            plt.subplot(nshow, 1, i + 1)
            plt.plot(self.wavelengths, self.evals[i] * ev, label="component: %d, %.2f"%(i, self.evals_cs[i]))
            if i == 0:
                plt.title('PCA Eigenspectra Phase %d$\pm$%d'%(self.loadPhase, self.phaseWidth), fontsize=18)
            if i == nshow - 1:
                plt.xlabel("Wavelength", fontsize=16)
            plt.ylabel("Rel Flux", fontsize=16)
            plt.legend(fontsize=12)
        return f

# Plot spectra

    def plotSpectra(self, figsize, alpha):
        f = plt.figure(figsize=figsize)
        for i, spec in enumerate(self.spectraMatrix):
            if not i % 10:
                plt.plot(self.wavelengths, spec + i*2, alpha=1.0)
            else:
                plt.plot(self.wavelengths, spec + i*2, alpha=alpha)
        plt.xlabel("Wavelengths (Angstroms)")
        plt.title("All Spectra")
        return f





#* SN Ia
#      typename(1,1) = 'Ia'      ! first element is name of type
#      typename(1,2) = 'Ia-norm' ! subtypes follow...(normal, peculiar, etc.)
#      typename(1,3) = 'Ia-91T'
#      typename(1,4) = 'Ia-91bg'
#      typename(1,5) = 'Ia-csm'
#      typename(1,6) = 'Ia-pec'
#      typename(1,7) = 'Ia-99aa'
#      typename(1,8) = 'Ia-02cx'

#* SN Ib      
#      typename(2,1) = 'Ib'
#      typename(2,2) = 'Ib-norm'
#      typename(2,3) = 'Ib-pec'
#      typename(2,4) = 'IIb'     ! IIb is not included in SNII
#      typename(2,5) = 'Ib-n'    ! Ib-n can be regarded as a kind of Ib-pec 

#* SN Ic
#      typename(3,1) = 'Ic'
#      typename(3,2) = 'Ic-norm'
#      typename(3,3) = 'Ic-pec'
#      typename(3,4) = 'Ic-broad'

#* SN II
#      typename(4,1) = 'II'
#      typename(4,2) = 'IIP'     ! IIP is the "normal" SN II
#      typename(4,3) = 'II-pec'
#      typename(4,4) = 'IIn'
#      typename(4,5) = 'IIL'

#* NotSN
#      typename(5,1) = 'NotSN'
#      typename(5,2) = 'AGN'
#      typename(5,3) = 'Gal'
#      typename(5,4) = 'LBV'
#      typename(5,5) = 'M-star'
#      typename(5,6) = 'QSO'
#      typename(5,7) = 'C-star'