# SESNspectraPCA/code/

This directory contains the code necessary to run the PCA and SVM spectral analysis presented in [Williamson & Modjaz & Bianco (2019)](https://arxiv.org/abs/1903.06815). The files here handle the following:

- <b>/PlotScripts</b> -- Contains scripts for generating each of the figures found in [Williamson & Modjaz & Bianco (2019)](https://arxiv.org/abs/1903.06815), as well as an additional plot comparing the first 5 eigenspectra across all four phases.
- <b>SNIDsn.py</b> -- Defines the SNIDsn class that is responsible for loading a single SNID .lnw template file.  
- <b>SNIDdataset.py</b> -- Defines functions for collecting multiple SNIDsn objects into a dictionary, and other functions for manipulating the entire dictionary during the PCA and SVM analysis.
- <b>SNePCA.py</b> -- Defines a SNePCA class for running the PCA and SVM analysis on a dataset of SNIDsn objects constructed using <b>SNIDdataset.py</b>.

In addition, this directory contains two Tutorial notebooks
- <b>SNIDdataset_SNIDsn_Tutorial.ipynb</b> -- A Jupyter Notebook that demonstrates how to use the SNIDsn class and SNIDdataset module to easily create your own SNID datasets.
- <b>Classify_New_SN_Tutorial.ipynb</b> -- A Jupyter Notebook that demonstrates how to classify a new supernova (SN2018gep is used as an example) using the method in [Williamson & Modjaz & Bianco (2019)](https://arxiv.org/abs/1903.06815).

And finally a notebook that reproduces the plots in [Williamson & Modjaz & Bianco (2019)](https://arxiv.org/abs/1903.06815).
- <b>Tutorial.ipynb</b> -- A Jupyter Notebook that uses the pickled SNID datasets in SESNspectraPCA/Data/DataProducts to generate the figures from [Williamson & Modjaz & Bianco (2019)](https://arxiv.org/abs/1903.06815). 

In order to run the Williamson_2019 Notebook (specifically to generate the eigenspectra vs mean spectra figure), you will need to download some of the mean spectra presented in [Liu et al. (2016)](http://adsabs.harvard.edu/abs/2016ApJ...827...90L) and [Modjaz et al. (2016)](http://adsabs.harvard.edu/abs/2016ApJ...832..108M). These mean spectra can be downloaded from the [nyusngroup github account](https://github.com/nyusngroup). You will need the following:

1. <b>meanspecIc_1specperSN_15_ft.sav</b> located [here](https://github.com/nyusngroup/SESNtemple/tree/master/MeanSpec/meanspIc/OneSpectrumPerSN)
2. <b>meanspecIcBL_1specperSN_15_ft.sav</b> located [here](https://github.com/nyusngroup/SESNtemple/tree/master/MeanSpec/meanspIcBL/OneSpectrumPerSN)
3. <b>meanspecIb_1specperSN_15.sav</b> located [here](https://github.com/nyusngroup/SESNtemple/tree/master/MeanSpec/meanspIb/OneSpectrumPerSN)
4. <b>meanspecIIb_1specperSN_15.sav</b> located [here](https://github.com/nyusngroup/SESNtemple/tree/master/MeanSpec/meanspIIb/OneSpectrumPerSN)

Collect the above mean spectra for the SESNe types at phase 15 days relative to V-band maximum in a directory, and define the following environmental variable:

`MEANSPEC=/path/to/meanspec/directory`



