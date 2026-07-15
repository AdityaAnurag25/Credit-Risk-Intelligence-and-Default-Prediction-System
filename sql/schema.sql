CREATE TABLE fact_loans (

    loan_id SERIAL PRIMARY KEY,

    loan_amnt FLOAT,
    funded_amnt FLOAT,
    term VARCHAR(20),
    int_rate FLOAT,
    installment FLOAT,

    grade VARCHAR(10),
    sub_grade VARCHAR(10),

    emp_length VARCHAR(50),
    home_ownership VARCHAR(50),

    annual_inc FLOAT,
    verification_status VARCHAR(50),

    purpose VARCHAR(100),

    dti FLOAT,

    revol_util FLOAT,

    delinq_2yrs FLOAT,
    inq_last_6mths FLOAT,

    open_acc FLOAT,
    pub_rec FLOAT,

    total_acc FLOAT,

    target_default INTEGER,

    income_to_loan_ratio FLOAT,
    installment_to_income FLOAT,
    credit_exposure_score FLOAT,

    risk_band VARCHAR(50)
);
