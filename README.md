# insidephy: Inter- and intraspecific Size Diversity of Phytoplankton

_insidephy_ is a package for modelling inter- and intraspecific 
size variability of phytoplankton grown under laboratory conditions.

# Installation

To install and utilize insidephy package a running distribution of Python (preferably 3.7 or above) is required. To install insidephy simply execute on a command line:

> $ pip install insidephy

Or download the tarball of the latest version of the package from the Github repository and install from source as:
 
> $ python setup.py install

# Usage

Once the package has been successfully installed on python, then specific modules of the insidephy package can be imported, for example, the SBMi model type as:

>from insidephy.size_based_models.SBMi import SBMi

To make a simulation first we need to define the initial conditions, notice that some values needs to be provided as a list:

> ini_resource = 0.002
> ini_density = [1e5, 1e10]
> min_size = [1.5e7, 1.5e10]
> max_size = [2.5e7, 2.5e10]
> spp_names = [‘Aa’, ‘Bb’]
> dilution_rate = 0.0
> volume = 1.0
> nsi_spp = [50]*2
> nsi_min = 200
> nsi_max = 2000
> time_end = 20
> time_step = 1 / (24*60)

Then to execute the simulation simply type:

> twospp = SBMi(ini_resource=ini_resource, ini_density=ini_density, minsize=min_size, maxsize=max_size, spp_names=spp_names, dilution_rate=dilution_rate, volume=volume, nsi_spp=nsi_spp, nsi_min=nsi_min, nsi_max=nsi_max, time_step=time_step, time_end=time_end)

The result of the execution will be stored as multidimensional arrays as part of the object twospp. Therefore, the results of the simulation can be accessed as instances of that object using the dot operator, like:

> twospp.resource
> twospp.biomass
> twospp.abundance
> twospp.quota
> twospp.agents_size
> twospp.agents_biomass
> twospp.agents_abundance







