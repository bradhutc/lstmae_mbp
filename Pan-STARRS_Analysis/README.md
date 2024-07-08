1. Go to https://mastweb.stsci.edu/mcasjobs/SubmitJob.aspx -----------------------------

2. Obtain the data from PS1-STRM: -----------------------------

SELECT objID, class, prob_Star
INTO mydb.MyTable
FROM HLSP_PS1_STRM.catalogRecordRowStore
WHERE class = 'STAR' AND b BETWEEN 60 AND 80;


3. Get photometry for PanStarrs -----------------------------

SELECT
    m.objID,
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
INTO mydb.PhotTable
FROM mydb.MyTable m
JOIN Panstarrs_dr1.MeanObject p ON m.objID = p.objID
WHERE 
    p.gMeanPSFMagErr <= 0.1 AND
    p.rMeanPSFMagErr <= 0.1 AND
    p.iMeanPSFMagErr <= 0.1 AND
    p.zMeanPSFMagErr <= 0.1 AND
    p.yMeanPSFMagErr <= 0.1


4. Get updated RA and DEC and Errors -----------------------------

SELECT
    m.objID,
    p.objID,
    p.raMean,
    p.decMean,
    p.raMeanErr,
    p.decMeanErr
INTO mydb.MatchedTable
FROM mydb.PhotTable m
JOIN Panstarrs_dr1.ObjectThin p ON m.objID = p.objID

5. Get GAIA Parallax and Parallax Errors ----------------------

SELECT
    m.objID,
    m.raMean AS ps_ra,
    m.decMean AS ps_dec,
    m.raMeanErr,
    m.decMeanErr,
    g.ra AS gaia_ra,
    g.dec AS gaia_dec,
    g.parallax,
    g.parallax_error,
    (
        SELECT COUNT(*)
        FROM GAIA_DR3.gaia_source g2
        WHERE
            g2.ra BETWEEN m.raMean - 2.5/3600.0 AND m.raMean + 2.5/3600.0
            AND g2.dec BETWEEN m.decMean - 2.5/3600.0 AND m.decMean + 2.5/3600.0
    ) AS nearby_object_count
INTO mydb.MatchedTableGaia
FROM mydb.MatchedTable m
JOIN GAIA_DR3.gaia_source g ON
    g.ra BETWEEN m.raMean - 2.5/3600.0 AND m.raMean + 2.5/3600.0
    AND g.dec BETWEEN m.decMean - 2.5/3600.0 AND m.decMean + 2.5/3600.0
WHERE
    m.raMeanErr < 1.0 AND m.decMeanErr < 1.0


6. Get u magnitude from SDSS -----------------------------

SELECT
    m.objID,
    m.prob_Star,
    m.raMean,
    m.decMean,
    m.raMeanErr,
    m.decMeanErr,
    s.psfMag_u,
    s.psfMagErr_u
INTO mydb.MatchedTableSDSS
FROM mydb.MatchedTable m
JOIN Star s ON
    s.ra BETWEEN m.raMean - 0.5/3600.0 AND m.raMean + 0.5/3600.0
    AND s.dec BETWEEN m.decMean - 0.5/3600.0 AND m.decMean + 0.5/3600.0
WHERE
    m.raMeanErr < 0.1 AND m.decMeanErr < 0.1
    AND (
        SELECT COUNT(*)
        FROM Star s2
        WHERE
            s2.ra BETWEEN m.raMean - 0.5/3600.0 AND m.raMean + 0.5/3600.0
            AND s2.dec BETWEEN m.decMean - 0.5/3600.0 AND m.decMean + 0.5/3600.0
    ) = 1

7. Get J, H, and K from 2MASS -----------------------------

SELECT
    m.objID,
    m.prob_Star,
    m.raMean,
    m.decMean,
    m.raMeanErr,
    m.decMeanErr,
    t.j_m AS jmag,
    t.j_cmsig AS jmag_err,
    t.h_m AS hmag,
    t.h_cmsig AS hmag_err,
    t.k_m AS kmag,
    t.k_cmsig AS kmag_err
INTO mydb.MatchedTable2MASS
FROM mydb.MatchedTable m
JOIN twomass_psc t ON
    t.ra BETWEEN m.raMean - 0.5/3600.0 AND m.raMean + 0.5/3600.0
    AND t.decl BETWEEN m.decMean - 0.5/3600.0 AND m.decMean + 0.5/3600.0
WHERE
    m.raMeanErr < 0.1 AND m.decMeanErr < 0.1
    AND t.j_snr >= 5 AND t.h_snr >= 5 AND t.k_snr >= 5
    AND (
        SELECT COUNT(*)
        FROM twomass_psc t2
        WHERE
            t2.ra BETWEEN m.raMean - 0.5/3600.0 AND m.raMean + 0.5/3600.0
            AND t2.decl BETWEEN m.decMean - 0.5/3600.0 AND m.decMean + 0.5/3600.0
    ) = 1
