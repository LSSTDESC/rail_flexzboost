# pz-rail-flexzboost

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/LSSTDESC/pz-rail-flexzboost/branch/main/graph/badge.svg)](https://codecov.io/gh/LSSTDESC/pz-rail-flexzboost)
[![PyPI](https://img.shields.io/pypi/v/flexzboost?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/flexzboost/)

## TODO List

To ensure that your project and repository is stable from the very start there 
are a few todo items that you should take care of. Unfortunately the RAIL project
template can not do these for you, otherwise it would :) 

### Immediate actions
- In your repository settings:
  -  Grant the `LSSTDESC/rail_admin` group administrator access
  -  Grant the `LSSTDESC/photo-z` group maintainer access
- Configure Codecov for the repository
  - Go here, https://github.com/apps/codecov, click the "Configure" button
- Log in to PyPI.org and configure Trusted Publishing following these instructions https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/
- Create a Personal Access Token (PAT) to automatically add issues to the RAIL project tracker
  - Follow these instruction to create a PAT: https://github.com/actions/add-to-project#creating-a-pat-and-adding-it-to-your-repository 
  - Save your new PAT as a repository secret named `ADD_TO_PROJECT_PAT`

### Before including in Rail-hub
- Make sure your `main` branch is protected
- Update this README
- Create an example notebook
- Run `pylint` on your code
- Remove this TODO list once all items are completed


## RAIL: Redshift Assessment Infrastructure Layers

This package is part of the larger ecosystem of Photometric Redshifts
in [RAIL](https://github.com/LSSTDESC/RAIL).

### Citing RAIL

This code, while public on GitHub, has not yet been released by DESC and is
still under active development. Our release of v1.0 will be accompanied by a
journal paper describing the development and validation of RAIL.

If you make use of the ideas or software in RAIL, please cite the repository 
<https://github.com/LSSTDESC/RAIL>. You are welcome to re-use the code, which
is open source and available under terms consistent with the MIT license.

External contributors and DESC members wishing to use RAIL for non-DESC projects
should consult with the Photometric Redshifts (PZ) Working Group conveners,
ideally before the work has started, but definitely before any publication or 
posting of the work to the arXiv.

### Citing this package

If you use this package, you should also cite the appropriate papers for each
code used.  A list of such codes is included in the 
[Citing RAIL](https://lsstdescrail.readthedocs.io/en/stable/source/citing.html)
section of the main RAIL Read The Docs page.

