# rail_flexzboost
RAIL interface to Flexzboost algorithms

The older version of FlexCode that is currently available via PyPI/`pip install` has several old keywords included that cause the code to run more slowly.  For optimal performance you will need to get the updated fork of FlexCode available from the Lee CMU group.   Unfortunately, that is currently not on PyPI, and PyPI packaging does not allow automated install from a GitHub repo direct link.  For optimal performance, after you install `rail_flexzboost`  run the command
```
pip install git+https://github.com/lee-group-cmu/FlexCode
```
to grab the forked version of FlexCode.  We hope to eliminate this extra step in the near future.
