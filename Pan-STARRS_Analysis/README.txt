Data obtained from https://archive.stsci.edu/hlsp/ps1-strm

Pan-STARRS1 Source Types and Redshifts with Machine Learning allows us to get just stars from the Pan-STARRS DR1 with little contamination from Galaxies and QSO's.

Appendix here: https://spacetelescope.github.io/hellouniverse/notebooks/hello-universe/Classifying_PanSTARRS_sources_with_unsupervised_learning/Classifying_PanSTARRS_sources_with_unsupervised_learning.html

1. Go to https://mastweb.stsci.edu/mcasjobs/SubmitJob.aspx

2. Obtain the data from PS1-STRM:

SELECT objID, raMean, decMean, l, b, prob_Star
INTO mydb.MyTable
FROM HLSP_PS1_STRM.catalogRecordRowStore
WHERE prob_Star > 0.99 AND b BETWEEN 60 AND 80;

3. Get RA and DEC errors.

SELECT
    m.objID,
    m.raMean,
    m.decMean,
    m.l,
    m.b,
    m.prob_Star,
    p.objID,
    p.raMeanErr,
    p.decMeanErr
INTO mydb.MatchedTable
FROM mydb.MyTable m
JOIN Panstarrs_dr1.ObjectThin p ON m.objID = p.objID

4. Get Photometry from Panstarrs_dr1

SELECT
    m.objID,
    m.raMean,
    m.raMeanErr,
    m.decMean,
    m.decMeanErr,
    m.l,
    m.b,
    m.prob_Star,
    p.gMeanPSFMag,
    p.gMeanPSFMagErr,
    p.gQfPerfect,
    p.gMeanPSFMagNpt,
    p.rMeanPSFMag,
    p.rMeanPSFMagErr,
    p.rQfPerfect,
    p.rMeanPSFMagNpt,
    p.iMeanPSFMag,
    p.iMeanPSFMagErr,
    p.iQfPerfect,
    p.iMeanPSFMagNpt,
    p.zMeanPSFMag,
    p.zMeanPSFMagErr,
    p.zQfPerfect,
    p.zMeanPSFMagNpt,
    p.yMeanPSFMag,
    p.yMeanPSFMagErr,
    p.yQfPerfect,
    p.yMeanPSFMagNpt
INTO mydb.FinalTable
FROM mydb.MatchedTable m
JOIN Panstarrs_dr1.MeanObject p ON m.objID = p.objID
