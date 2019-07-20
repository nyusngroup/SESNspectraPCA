# SESNspectraPCA

This repository contains the code for running the PCA + SVM spectral analysis on Stripped-Envelope Core-Collapse Supernovae found in [Williamson & Modjaz & Bianco (2019)](https://arxiv.org/abs/1903.06815). There are also data products associated with the new SNID templates that were added to the SESNe spectral library compiled from [Liu et al. 2016](http://adsabs.harvard.edu/abs/2016ApJ...827...90L) and [Modjaz et al. 2016](http://adsabs.harvard.edu/abs/2016ApJ...832..108M). The repository is organized into the following folders:
- <b>/Data</b> contains the pickled spectral datasets used in [Williamson & Modjaz & Bianco (2019)](https://arxiv.org/abs/1903.06815) at phases 0, 5, 10, 15 days relative to V-band maximum. The spectra have been preprocessed. These pickled files are used in the Tutorial notebook.
- <b>/SNePCAplots</b> contains final figures showing PCA reconstruction of SN2011ei, cumulative variance captured by principal components, time evolution of the eigenspectra, comparison of mean spectra vs eigenspectra, and SVM classification of SESNe.
- <b>/code </b> contains the SNePCA.py which implements the PCA and SVM analysis used in [Williamson & Modjaz & Bianco (2019)](https://arxiv.org/abs/1903.06815). It also contains SNIDsn.py and SNIDdataset.py which are used to load the SNID template files (\*.lnw files) and compile them into a dataset for easy use. There are tutorial jupyter notebooks, which generates plots from [Williamson & Modjaz & Bianco (2019)](https://arxiv.org/abs/1903.06815) and demonstrates how the code is run.

# Installing SESNspectraPCA

In order to install the code, you will need [conda](https://docs.conda.io/en/latest/miniconda.html). Clone or Fork the SESNspectraPCA repository (we encourage pull requests!) then install the conda virtual environment:

`conda env create -f sesn_env.yml`

The code only supports Python 3, and pickled dataproducts use protocol=2 which is compatible for Python 3 and Python 2. Check out the [tutorial notebook](https://github.com/nyusngroup/SESNspectraPCA/tree/master/code) to get started.

# How to Use the Code

- <b>Classify new SN spectra</b> -- You may want to use the classification method from [Williamson & Modjaz & Bianco (2019)](https://arxiv.org/abs/1903.06815) to classify your own spectra. To get started, see the Classify_New_SN_Tutorial.ipynb notebook in the /code directory.
- <b>Create your own SNID dataset</b> -- If you are doing your own research using SNID templates, you may find the functionality of the SNIDsn class and SNIDdataset module extremely helpful. To get started using these resources, see the SNIDdataset_SNIDsn_Tutorial.ipynb notebook in the /code directory.

# Compiling the SESNe SNID Dataset

If you are interested in conducting your own research on stripped-envelope supernovae, you may want to compile our entire SESNe SNID template dataset. This includes the dataset of SuperNova IDentification (SNID); [Blondin & Tonry 2007](https://iopscience.iop.org/article/10.1086/520494/meta) as well as new and modified SNID templates from our group. Follow these steps:

1. <b>Download the default SNID library</b> -- The default library of SESNe spectra for SNID can be found on Stephan Blondin's [website](https://people.lam.fr/blondin.stephane/software/snid/)
2. <b>Add SNID Templates from [nyusngroup](https://github.com/nyusngroup/SESNtemple)</b> -- The research group at New York University (NYU) lead by Professor Maryam Modjaz has published multiple papers presenting new SNID templates of SESNe in the literature, along with making adjustments to the SNID templates in the default SNID library. [SESNtemple](https://github.com/nyusngroup/SESNtemple) contains the new and adjusted SNID templates from [Liu & Modjaz 2014](http://adsabs.harvard.edu/abs/2014arXiv1405.1437L), [Liu et al. 2016](http://adsabs.harvard.edu/abs/2016ApJ...827...90L) and in [Modjaz et al. (2016)](http://adsabs.harvard.edu/abs/2016ApJ...832..108M), that are based on both CfA Data (initially released in Liu & Modjaz 2014) and the rest of literature data (included in Liu et al. 2016 for IIb and Ib, Modjaz et al. 2016 for Ic and Ic-bl). There are also SNID templates for the superluminous SNe presented in [Liu, Modjaz & Bianco (2017)](http://adsabs.harvard.edu/abs/2016arXiv161207321L), based on literature data. Finally, the new SNID templates for SESNe in the literature through August 2018 presented in [Williamson & Modjaz & Bianco (2019)](https://arxiv.org/abs/1903.06815) are included.
3. <b>Set up SNID Template Directory</b> -- Gather all of the SNID templates from the above two steps in a directory, and consider defining an environmental variable to be the path to your directory.

# Acknowledgement:

If you use data products or the analysis in this code, please <b>acknowledge</b> this work by citing in your paper: [Williamson & Modjaz & Bianco (2019)](https://arxiv.org/abs/1903.06815).

Williamson & Modjaz & Bianco (2019):

  	@article{1903.06815,
    Author = {Marc Williamson and Maryam Modjaz and Federica Bianco},
    Title = {Optimal Classification and Outlier Detection for Stripped-Envelope Core-Collapse Supernovae},
    Year = {2019},
    Eprint = {arXiv:1903.06815},
    }
