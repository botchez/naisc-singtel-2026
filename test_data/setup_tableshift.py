"""Download and prepare TableShift benchmark datasets for testing the NAISC pipeline.

Each dataset gets a train.csv and test.csv with a real-world distribution shift:
  1. acsincome        — Income prediction, shift by US state (mild)
  2. diabetes         — Hospital readmission, shift by admission source (moderate)
  3. acspubcov        — Public health insurance, shift by US state (severe)
  4. acsunemployment  — Unemployment prediction, shift by US state
  5. acsfoodstamps    — Food stamp recipiency, shift by US state (household+person merge)
  6. college_scorecard— Low degree completion, shift by institution control type
  7. brfss_diabetes   — Diabetes diagnosis, shift by US state (CDC BRFSS survey)
  8. brfss_blood_pressure — Hypertension, shift by US state (CDC BRFSS survey)
  9. nhanes_lead      — Childhood lead exposure, shift by ethnicity (CDC NHANES)

Usage:
    python test_data/setup_tableshift.py           # download all
    python test_data/setup_tableshift.py acsincome  # download one

Output:
    test_data/<name>/train.csv, test_data/<name>/test.csv
"""

import os
import sys
import zipfile
import io
import urllib.request

import numpy as np
import pandas as pd
import polars as pl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Helpers ──────────────────────────────────────────────────────────────

def _split_by_column(df, col, ood_val, sample_train=50000, sample_test=12500):
    """Split df into train/test by column value, with optional downsampling."""
    if col not in df.columns:
        print(f"  Warning: '{col}' not found, using random 80/20 split")
        np.random.seed(42)
        n = len(df)
        mask = np.random.rand(n) < 0.8
        train_df = df.filter(pl.Series(mask))
        test_df = df.filter(~pl.Series(mask))
    else:
        unique_vals = df[col].drop_nulls().unique().sort().to_list()
        col_values = df[col].to_list()
        if ood_val not in col_values:
            # Try numeric coercion
            for cast in [float, int]:
                try:
                    v = cast(ood_val)
                    if v in col_values:
                        ood_val = v
                        break
                except (ValueError, TypeError):
                    pass
            else:
                ood_val = unique_vals[-1]
        print(f"  Split: {col} != {ood_val} (train) vs == {ood_val} (test)")
        train_df = df.filter(pl.col(col) != ood_val)
        test_df = df.filter(pl.col(col) == ood_val)

    if len(train_df) > sample_train:
        train_df = train_df.sample(n=sample_train, seed=42)
    if len(test_df) > sample_test:
        test_df = test_df.sample(n=sample_test, seed=42)
    return train_df, test_df


def _save(train_df, test_df, out_dir, name):
    """Save train/test CSVs and print summary."""
    os.makedirs(out_dir, exist_ok=True)
    train_df.write_csv(os.path.join(out_dir, "train.csv"))
    test_df.write_csv(os.path.join(out_dir, "test.csv"))
    print(f"  [{name}] Saved: train={train_df.shape}, test={test_df.shape}")
    print(f"  Target rate: train={train_df['target'].mean():.3f}, test={test_df['target'].mean():.3f}")


# ── 1. ACS datasets (folktables) ────────────────────────────────────────

def _get_acs_data(states=None):
    """Download ACS person-level data."""
    from folktables import ACSDataSource
    if states is None:
        states = ["CA", "NY", "TX", "FL", "IL"]
    ds = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    return ds.get_data(states=states, download=True)


def setup_acs_dataset(task_name, out_dir):
    """Download ACS data via folktables and split by state for drift."""
    from folktables import ACSIncome, ACSPublicCoverage, ACSEmployment

    task_map = {
        "acsincome": ACSIncome,
        "acspubcov": ACSPublicCoverage,
        "acsunemployment": ACSEmployment,
    }
    # OOD state: FL (state code 12) — different region from CA/NY/TX/IL
    ood_state = 12

    task = task_map[task_name]
    print(f"\n[{task_name}] Downloading ACS data (2018, 1-year)...")
    acs_data = _get_acs_data()

    features_pd, label_pd, group = task.df_to_pandas(acs_data)
    df = pl.from_pandas(features_pd)

    # folktables already applies the target transform (returns bool)
    df = df.with_columns(pl.Series("target", label_pd.astype(int).values))

    # Carry over ST (state) for splitting
    if "ST" not in df.columns and "ST" in acs_data.columns:
        st_values = acs_data["ST"].values[:len(df)]
        df = df.with_columns(pl.Series("ST", st_values))

    sample_ids = pl.DataFrame({"SampleID": [f"S{i:06d}" for i in range(len(df))]})
    df = pl.concat([sample_ids, df], how="horizontal")

    train_df, test_df = _split_by_column(df, "ST", ood_state)
    _save(train_df, test_df, out_dir, task_name)


def setup_acsfoodstamps(out_dir):
    """ACS Food Stamps: merge person + household data, split by state."""
    from folktables import ACSDataSource

    print("\n[acsfoodstamps] Downloading ACS person + household data (2018)...")
    states = ["CA", "NY", "TX", "FL", "IL"]

    # Person data
    ds_p = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    person_pd = ds_p.get_data(states=states, download=True)

    # Household data (has FS and HUPAC columns)
    ds_h = ACSDataSource(survey_year="2018", horizon="1-Year", survey="household")
    household_pd = ds_h.get_data(states=states, download=True)

    # Keep only needed household columns and merge on SERIALNO
    hh_cols = ["SERIALNO"]
    for c in ["FS", "HUPAC"]:
        if c in household_pd.columns:
            hh_cols.append(c)
    hh_subset = pl.from_pandas(household_pd[hh_cols]).unique(subset=["SERIALNO"])

    person = pl.from_pandas(person_pd)
    df = person.join(hh_subset, on="SERIALNO", how="left")

    # Filter: working age, at least one child, low income (TableShift criteria)
    df = df.filter((pl.col("AGEP") >= 18) & (pl.col("AGEP") < 62))
    if "HUPAC" in df.columns:
        df = df.filter(pl.col("HUPAC") >= 1)
    if "PINCP" in df.columns:
        df = df.filter(pl.col("PINCP") <= 30000)

    # Target: FS == 1 means receives food stamps
    df = df.with_columns((pl.col("FS") == 1).cast(pl.Int64).alias("target"))

    # Select features (subset of useful person-level + household features)
    feature_cols = [c for c in ["AGEP", "SCHL", "MAR", "SEX", "DIS", "ESP",
                                "MIG", "CIT", "ANC", "NATIVITY", "DEAR",
                                "DEYE", "DREM", "RAC1P", "RELP", "MIL",
                                "PINCP", "WKHP", "JWMNP", "HUPAC", "ST"]
                    if c in df.columns]
    out = df.select(feature_cols + ["target"])
    sample_ids = pl.DataFrame({"SampleID": [f"F{i:06d}" for i in range(len(out))]})
    out = pl.concat([sample_ids, out], how="horizontal")

    train_df, test_df = _split_by_column(out, "ST", 12)
    _save(train_df, test_df, out_dir, "acsfoodstamps")


# ── 2. Diabetes readmission (UCI) ───────────────────────────────────────

def setup_diabetes_readmission(out_dir):
    """Download UCI diabetes readmission dataset and split by admission_source_id."""
    url = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
    print("\n[diabetes] Downloading from UCI...")

    try:
        response = urllib.request.urlopen(url, timeout=60)
        zip_data = response.read()
    except Exception as e:
        print(f"  Error downloading: {e}")
        print("  Trying alternative URL...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
        response = urllib.request.urlopen(url, timeout=60)
        zip_data = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
        csv_names = [n for n in z.namelist() if n.endswith(".csv") and "diabetic" in n.lower()]
        if not csv_names:
            csv_names = [n for n in z.namelist() if n.endswith(".csv")]
        csv_name = csv_names[0]
        print(f"  Reading {csv_name}...")
        with z.open(csv_name) as f:
            df = pl.read_csv(f)

    print(f"  Raw shape: {df.shape}")

    if "readmitted" in df.columns:
        df = df.with_columns(
            (pl.col("readmitted") == "<30").cast(pl.Int64).alias("target")
        )
        df = df.drop(["readmitted"])
    else:
        last_col = df.columns[-1]
        df = df.rename({last_col: "target"})

    # Replace "?" with null across all string columns
    str_cols = [c for c in df.columns if df[c].dtype == pl.Utf8]
    if str_cols:
        df = df.with_columns([
            pl.when(pl.col(c) == "?").then(None).otherwise(pl.col(c)).alias(c)
            for c in str_cols
        ])

    if "encounter_id" in df.columns:
        df = df.rename({"encounter_id": "SampleID"})
    else:
        sample_ids = pl.DataFrame({"SampleID": [f"D{i:06d}" for i in range(len(df))]})
        df = pl.concat([sample_ids, df], how="horizontal")

    train_df, test_df = _split_by_column(df, "admission_source_id", 7,
                                         sample_train=60000, sample_test=15000)
    _save(train_df, test_df, out_dir, "diabetes")


# ── 3. College Scorecard ────────────────────────────────────────────────

def setup_college_scorecard(out_dir):
    """Download College Scorecard data via API and split by institution control type."""
    import requests as _req

    print("\n[college_scorecard] Downloading via College Scorecard API...")

    # API fields matching TableShift features
    api_fields = [
        "id", "school.name", "school.state", "school.ownership",
        "school.degrees_awarded.highest", "school.region_id", "school.locale",
        "admissions.admission_rate.overall",
        "admissions.sat_scores.midpoint.critical_reading",
        "admissions.sat_scores.midpoint.math",
        "admissions.act_scores.midpoint.cumulative",
        "student.size", "student.enrollment.undergrad_12_month",
        "cost.avg_net_price.overall", "cost.tuition.in_state",
        "cost.tuition.out_of_state", "school.tuition_revenue_per_fte",
        "school.instructional_expenditure_per_fte",
        "school.faculty_salary", "school.ft_faculty_rate",
        "aid.pell_grant_rate", "aid.federal_loan_rate",
        "student.demographics.race_ethnicity.white",
        "student.demographics.race_ethnicity.black",
        "student.demographics.race_ethnicity.hispanic",
        "student.demographics.race_ethnicity.asian",
        "school.online_only",
        "school.minority_serving.historically_black",
        "completion.rate_suppressed.overall",
    ]

    all_rows = []
    page = 0
    per_page = 100
    while True:
        url = (f"https://api.data.gov/ed/collegescorecard/v1/schools.json"
               f"?api_key=DEMO_KEY&per_page={per_page}&page={page}"
               f"&fields={','.join(api_fields)}")
        try:
            r = _req.get(url, timeout=30)
            data = r.json()
        except Exception as e:
            print(f"  API error on page {page}: {e}")
            break

        results = data.get("results", [])
        if not results:
            break
        all_rows.extend(results)
        total = data.get("metadata", {}).get("total", 0)
        print(f"  Page {page}: {len(results)} schools (total: {len(all_rows)}/{total})")
        if len(all_rows) >= total:
            break
        page += 1

    if not all_rows:
        print("  No data from API, skipping")
        return

    # Flatten nested JSON — no polars equivalent, use pandas then convert
    df_pandas = pd.json_normalize(all_rows)
    df = pl.from_pandas(df_pandas)
    print(f"  Raw shape: {df.shape}")

    # Target: completion rate, binarized at median
    comp_col = "completion.rate_suppressed.overall"
    if comp_col not in df.columns:
        print("  Warning: completion rate not found, skipping")
        return

    df = df.with_columns(pl.col(comp_col).cast(pl.Float64, strict=False))
    df = df.drop_nulls(subset=[comp_col])
    median_rate = df[comp_col].median()
    df = df.with_columns(
        (pl.col(comp_col) < median_rate).cast(pl.Int64).alias("target")
    )
    df = df.drop([comp_col])

    # Rename columns to shorter names
    rename_map = {
        "id": "SampleID", "school.ownership": "CONTROL",
        "school.degrees_awarded.highest": "HIGHDEG",
        "school.region_id": "region", "school.locale": "LOCALE",
        "admissions.admission_rate.overall": "ADM_RATE",
        "admissions.sat_scores.midpoint.critical_reading": "SATVRMID",
        "admissions.sat_scores.midpoint.math": "SATMTMID",
        "admissions.act_scores.midpoint.cumulative": "ACTCMMID",
        "student.size": "UGDS",
        "cost.tuition.in_state": "TUITIONFEE_IN",
        "cost.tuition.out_of_state": "TUITIONFEE_OUT",
        "school.tuition_revenue_per_fte": "TUITFTE",
        "school.instructional_expenditure_per_fte": "INEXPFTE",
        "school.faculty_salary": "AVGFACSAL",
        "school.ft_faculty_rate": "PFTFAC",
        "aid.pell_grant_rate": "PCTPELL",
        "aid.federal_loan_rate": "PCTFLOAN",
        "student.demographics.race_ethnicity.white": "UGDS_WHITE",
        "student.demographics.race_ethnicity.black": "UGDS_BLACK",
        "student.demographics.race_ethnicity.hispanic": "UGDS_HISP",
        "student.demographics.race_ethnicity.asian": "UGDS_ASIAN",
        "school.online_only": "DISTANCEONLY",
        "school.minority_serving.historically_black": "HBCU",
    }
    # Only rename columns that exist in df
    existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    out = df.rename(existing_renames)

    # Keep only renamed columns + target
    keep = [v for v in rename_map.values() if v in out.columns] + ["target"]
    keep = [c for c in keep if c in out.columns]
    out = out.select(keep)

    # Convert numeric columns
    for c in out.columns:
        if c not in ["SampleID", "target"]:
            out = out.with_columns(pl.col(c).cast(pl.Float64, strict=False))

    # Split: CONTROL == 3 (for-profit) as OOD
    train_df, test_df = _split_by_column(out, "CONTROL", 3,
                                         sample_train=50000, sample_test=12500)
    _save(train_df, test_df, out_dir, "college_scorecard")


# ── 4. BRFSS (CDC Behavioral Risk Factor Surveillance System) ───────────

def _download_brfss_year(year):
    """Download a single year of BRFSS data (SAS transport format)."""
    import requests as _req

    urls = [
        f"https://www.cdc.gov/brfss/annual_data/{year}/files/LLCP{year}XPT.zip",
        f"https://www.cdc.gov/brfss/annual_data/{year}/files/LLCP{str(year)[-2:]}XPT.zip",
    ]
    zip_data = None
    for url in urls:
        print(f"  Downloading BRFSS {year} from {url}...")
        try:
            r = _req.get(url, timeout=300, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and len(r.content) > 1000:
                zip_data = r.content
                print(f"    Downloaded {len(zip_data) / 1e6:.1f} MB")
                break
        except Exception as e:
            print(f"    Failed: {e}")

    if zip_data is None:
        print(f"    Could not download BRFSS {year}")
        return None

    with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
        xpt_names = [n for n in z.namelist() if n.strip().upper().endswith(".XPT")]
        if not xpt_names:
            print(f"    No XPT file found for {year}, files: {z.namelist()}")
            return None
        xpt_name = xpt_names[0]
        with z.open(xpt_name) as f:
            xpt_data = f.read()

    # pd.read_sas has no polars equivalent — read with pandas, then convert
    df_pandas = pd.read_sas(io.BytesIO(xpt_data), format="xport")
    df_pandas["IYEAR"] = year
    df = pl.from_pandas(df_pandas)
    return df


def setup_brfss(task, out_dir):
    """Download BRFSS data and create diabetes or blood pressure prediction dataset."""
    print(f"\n[brfss_{task}] Downloading CDC BRFSS data...")

    # Use 2021 (most recent available) - single year for simplicity
    years = [2021]
    dfs = []
    for year in years:
        ydf = _download_brfss_year(year)
        if ydf is not None:
            dfs.append(ydf)

    if not dfs:
        print("  Failed to download BRFSS data, skipping")
        return

    df = pl.concat(dfs)
    print(f"  Raw shape: {df.shape}")

    # Shared features across BRFSS tasks
    shared_features = ['CHECKUP1', 'CVDSTRK3', 'EDUCA', 'EMPLOY1',
                       'MARITAL', 'MEDCOST', 'MENTHLTH', 'PHYSHLTH',
                       'SEX', 'SMOKDAY2', 'SMOKE100',
                       '_AGEG5YR', '_BMI5', '_BMI5CAT', '_MICHD',
                       '_PRACE1', '_RFBING5', '_STATE', '_TOTINDA']

    if task == "diabetes":
        # DIABETE4 (or DIABETE3 in older years): diabetes status
        target_col = None
        for c in ["DIABETE4", "DIABETE3", "DIABETES"]:
            if c in df.columns:
                target_col = c
                break
        if target_col is None:
            print("  Warning: diabetes column not found, skipping")
            return

        extra_features = ['INCOME3', 'ADDEPEV3', 'BPHIGH6', '_RFHYPE6',
                          'TOLDHI3', '_CHOLCH2', '_RFCHOL3', 'HAVARTH4',
                          'RENTEFLO', 'ASTHMA3']
        # Binarize: 1 = diagnosed diabetes, 0 = no diabetes
        df = df.with_columns(
            pl.col(target_col).replace({2: 0, 3: 0, 4: 0}).alias("target")
        )
        df = df.filter(pl.col("target").is_in([0, 1]))

    elif task == "blood_pressure":
        target_col = None
        for c in ["BPHIGH6", "BPHIGH4"]:
            if c in df.columns:
                target_col = c
                break
        if target_col is None:
            print("  Warning: blood pressure column not found, skipping")
            return

        extra_features = ['INCOME3', 'ADDEPEV3', 'DIABETE4', 'DIABETE3',
                          'TOLDHI3', '_CHOLCH2', 'HAVARTH4', 'ASTHMA3']
        # BPHIGH6: 1=Yes, 2=Yes (pregnancy), 3=No, 4=Borderline
        # Binarize: 1,2 -> 1 (high BP), 3 -> 0 (no), drop 4,7,9
        df = df.with_columns(
            pl.col(target_col).replace({1: 1, 2: 1, 3: 0}).alias("target")
        )
        df = df.filter(pl.col("target").is_in([0, 1]))

        # Filter to age 50+ (highest risk group, per TableShift)
        if "_AGEG5YR" in df.columns:
            df = df.filter(pl.col("_AGEG5YR") >= 7)
    else:
        print(f"  Unknown BRFSS task: {task}")
        return

    # Select available features
    all_features = shared_features + extra_features
    available = [c for c in all_features if c in df.columns]
    out = df.select(available + ["target"])
    sample_ids = pl.DataFrame({"SampleID": [f"B{i:06d}" for i in range(len(out))]})
    out = pl.concat([sample_ids, out], how="horizontal")

    # Split by state (_STATE): OOD = 6 (California)
    train_df, test_df = _split_by_column(out, "_STATE", 6.0,
                                         sample_train=60000, sample_test=15000)
    _save(train_df, test_df, out_dir, f"brfss_{task}")


# ── 5. NHANES Lead ──────────────────────────────────────────────────────

def setup_nhanes_lead(out_dir):
    """Download NHANES data for childhood lead exposure prediction."""
    print("\n[nhanes_lead] Downloading CDC NHANES data...")

    import requests as _req

    # Download demographics + lab data for recent cycles
    # CDC requires specific URL pattern for raw XPT files
    cycles = [
        ("2017", "2017-2018", "DEMO_J.XPT", "PBCD_J.XPT"),
        ("2015", "2015-2016", "DEMO_I.XPT", "PBCD_I.XPT"),
    ]

    def _fetch_xpt(year, cycle, filename):
        url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{filename}"
        r = _req.get(url, timeout=60)
        if r.status_code == 200 and r.content[:6] == b"HEADER":
            # pd.read_sas has no polars equivalent
            return pl.from_pandas(pd.read_sas(io.BytesIO(r.content), format="xport"))
        # Fallback URL
        url2 = f"https://wwwn.cdc.gov/Nchs/Nhanes/{cycle}/{filename}"
        r2 = _req.get(url2, timeout=60)
        if r2.status_code == 200 and r2.content[:6] == b"HEADER":
            return pl.from_pandas(pd.read_sas(io.BytesIO(r2.content), format="xport"))
        raise ValueError(f"Could not download {filename} for {cycle}")

    dfs = []
    for year, cycle, demo_file, lab_file in cycles:
        print(f"  Downloading {cycle}...")
        try:
            demo = _fetch_xpt(year, cycle, demo_file)
            lab = _fetch_xpt(year, cycle, lab_file)

            merged = demo.join(lab, on="SEQN", how="inner")
            merged = merged.with_columns(pl.lit(cycle).alias("CYCLE"))
            dfs.append(merged)
            print(f"    {cycle}: {merged.shape}")
        except Exception as e:
            print(f"    Failed {cycle}: {e}")

    if not dfs:
        print("  Failed to download NHANES data, skipping")
        return

    df = pl.concat(dfs)
    print(f"  Combined shape: {df.shape}")

    # Target: blood lead level >= 3.5 µg/dL (CDC reference value)
    lead_col = None
    for c in ["LBXBPB", "LBDBPBSI"]:
        if c in df.columns:
            lead_col = c
            break
    if lead_col is None:
        print("  Warning: lead column not found, skipping")
        return

    df = df.with_columns(pl.col(lead_col).cast(pl.Float64, strict=False))
    df = df.drop_nulls(subset=[lead_col])
    df = df.with_columns(
        (pl.col(lead_col) >= 3.5).cast(pl.Int64).alias("target")
    )

    # Select features
    feature_candidates = ["RIDAGEYR", "RIAGENDR", "RIDRETH3", "RIDRETH1",
                          "DMDEDUC2", "DMDMARTL", "INDFMPIR",
                          "RIDEXPRG", "DMDBORN4", "DMDCITZN",
                          "WTMEC2YR", "SDMVSTRA"]
    features = [c for c in feature_candidates if c in df.columns]
    out = df.select(features + ["target"])
    sample_ids = pl.DataFrame({"SampleID": [f"N{i:06d}" for i in range(len(out))]})
    out = pl.concat([sample_ids, out], how="horizontal")

    # Split by ethnicity (RIDRETH3 or RIDRETH1): OOD = 4 (Non-Hispanic Asian)
    split_col = "RIDRETH3" if "RIDRETH3" in out.columns else "RIDRETH1"
    train_df, test_df = _split_by_column(out, split_col, 6.0,
                                         sample_train=50000, sample_test=12500)
    _save(train_df, test_df, out_dir, "nhanes_lead")


# ── Main ─────────────────────────────────────────────────────────────────

ALL_DATASETS = [
    "acsincome", "acspubcov", "acsunemployment", "acsfoodstamps",
    "diabetes",
    "brfss_diabetes", "brfss_blood_pressure",
    "nhanes_lead",
    "college_scorecard",  # requires API key, may hit rate limits
]


def setup_one(name):
    out_dir = os.path.join(SCRIPT_DIR, name)

    if name in ("acsincome", "acspubcov", "acsunemployment"):
        setup_acs_dataset(name, out_dir)
    elif name == "acsfoodstamps":
        setup_acsfoodstamps(out_dir)
    elif name == "diabetes":
        setup_diabetes_readmission(out_dir)
    elif name == "college_scorecard":
        setup_college_scorecard(out_dir)
    elif name == "brfss_diabetes":
        setup_brfss("diabetes", out_dir)
    elif name == "brfss_blood_pressure":
        setup_brfss("blood_pressure", out_dir)
    elif name == "nhanes_lead":
        setup_nhanes_lead(out_dir)
    else:
        print(f"Unknown dataset: {name}")


def main():
    print("=" * 60)
    print("Setting up TableShift benchmark datasets")
    print("=" * 60)

    # Allow selecting specific datasets via CLI args
    if len(sys.argv) > 1:
        datasets = [d for d in sys.argv[1:] if d in ALL_DATASETS]
        if not datasets:
            print(f"Unknown dataset(s). Available: {ALL_DATASETS}")
            return
    else:
        datasets = ALL_DATASETS

    for name in datasets:
        try:
            setup_one(name)
        except Exception as e:
            print(f"\n  ERROR setting up {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Done! Run the pipeline on each dataset:")
    print()
    for name in datasets:
        d = os.path.join("test_data", name)
        if os.path.exists(os.path.join(SCRIPT_DIR, name, "train.csv")):
            print(f"  python src/main.py --train_data_filepath {d}/train.csv --test_data_filepath {d}/test.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
