-- Portfolio Exposure Summary
SELECT
    risk_band,
    COUNT(*) AS total_loans,
    ROUND(
        SUM(loan_amnt),
        2
    ) AS total_exposure,
    ROUND(
        AVG(int_rate),
        2
    ) AS avg_interest_rate
FROM fact_loans
GROUP BY risk_band
ORDER BY total_exposure DESC;

-- DTI Risk Segmentation
SELECT
    CASE
        WHEN dti < 10 THEN 'Low DTI'
        WHEN dti < 20 THEN 'Moderate DTI'
        WHEN dti < 30 THEN 'High DTI'
        ELSE 'Very High DTI'
    END AS dti_segment,

    COUNT(*) AS borrower_count,

    ROUND(
        AVG(target_default) * 100,
        2
    ) AS default_rate

FROM fact_loans

GROUP BY dti_segment

ORDER BY default_rate DESC;

