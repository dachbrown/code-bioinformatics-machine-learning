by David Brown (db) 20221208

README for Basic R Workspace in VS Code

** PURPOSE **
This method should allow for multiple R environments per project as needed, when managed using Anaconda.
*****

** INSTRUCTIONS **
After installing the below packages (in a clean Anaconda environment for R):
 - r-base               ==  base R
 - r-phytools           ==  basic phylogenetic tool set in R
 - r-ape                ==  another basic phylogenetic tool set in R
 - r-languageserver     ==  required to make R work with VS Code
 - radian               ==  suggested R terminal
 - r-httpgd             ==  suggested R package for graphics that works with VS Code
 - r-feather            ==  feather format for R
 - feather-format       ==  feather format for Python
 - r-dplyr              ==  standardize data manipulation in R

### NOTE ###
Both Python and R can read feather format files. These installation instructions will
automatically install Python, as it is a dependency for "radian". The calls for both
feather packages from Anaconda ensure that Python code run in this environment should
at least TEMPORARILY work with R code.
### NOTE ###

Follow the remaining instructions at https://code.visualstudio.com/docs/languages/r,
specifically focus on the 'R extension for Visual Studio Code'.

Make certain to explicitly set the paths for both R and the R terminal within the extension
settings for your specific machine (Linux, MacOS, Windows).
*****

** ISSUES **
It is possible you will run into issues, especially when launching a new interactive terminal.
Setting your paths for R and the R terminal "radian" could help.
Not necessarily best practice, but a working solution is to:
 1) Select "R: Always Use Active Terminal" in the extension settings for R.
 2) Manually open a new terminal in VS Code
 3) Activate the correct Anaconda environment
 4) Enter "radian --no-save --no-restore" into the terminal
The above should allow the terminal to source your code in the open terminal upon selecting the "Run" button arrow.

LOG
- started 1955 20221209