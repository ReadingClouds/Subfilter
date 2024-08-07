# Subfilter
python code to compute sub-filter quantities from MONC output.
See https://readingclouds.github.io/Subfilter/ for documentation.

Current version: 0.6.1.

Note this is a major re-organisation, with the packages io, thermodynamics and 
utils moved to https://github.com/ReadingClouds/monc_utils.git. Please
ensure this is also installed. See the README and documentation for that 
repository.

Users should pip install to a suitable environment using

    pip install  git+https://github.com/ReadingClouds/Subfilter.git

This will install into the standard library.

Developers should fork then clone the repository (please create a branch before making 
any changes!), open a terminal window and activate the python environment 
required, cd to the Subfilter directory and

    pip install -e .

This will install as if into the standard library but using the cloned code 
which can be edited. Please commit code improvements and discuss merging with 
the master branch with Peter Clark and other users.
