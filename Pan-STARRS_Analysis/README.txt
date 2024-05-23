Data obtained from https://archive.stsci.edu/hlsp/ps1-strm

Pan-STARRS1 Source Types and Redshifts with Machine Learning allows us to get just stars from the Pan-STARRS DR1 with little contamination from Galaxies and QSO's.

Appendix here: https://spacetelescope.github.io/hellouniverse/notebooks/hello-universe/Classifying_PanSTARRS_sources_with_unsupervised_learning/Classifying_PanSTARRS_sources_with_unsupervised_learning.html

-- 0_obtainstars.py will take the PS1-STRM file and get stars with a desired probability level. 
-- 1_obtain_phot will match using objID to get photometric measurements of the stars in DR1 (griz) along with their uncertainties.
