import pandas as pd
import pyodbc
import re

# ── Database connection ──────────────────────────────────────────────────────
conn = pyodbc.connect("DSN=DEVO Impala 64bit")

query = """
    SELECT srep_cycle, name, jst_code, ssm_dg, ssm_div, ssm_section, cluster,
           element_shortname, element_internalidentifier, element_fullname,
           element_block_code, element_risk, element_type, phase, element_module,
           element_module_name, element_module_level, latest_score,
           scopeofassessmenttype, assessmentreason, engagement_name, cons_level
    FROM crp_agora.agora_dm_srep_plan
    WHERE top_consolidation='Y'
      AND srep_cycle IN ('2023','2024','2025')
      AND ssm_div NOT IN ('DIV Institutional & Sectoral Oversight', 'NULL')
      AND ssm_section IS NOT NULL
      AND engagement_name LIKE 'SREP%'
    ORDER BY jst_code, entity_id, cons_level, 'element', element_risk, element_module
"""
MYA = pd.read_sql(query, conn)

# ── Load Excel files ─────────────────────────────────────────────────────────
base = "C:/Users/depaola/Desktop/Dashboard work/"

risk_areas  = pd.read_excel(base + "Risk_area_RTF.xlsx")
RTF_Extract = pd.read_excel(base + "RTF-SSPs.xlsx")
OSIs        = pd.read_excel(base + "OSIs.xlsx", sheet_name="OSI")
HorAct      = pd.read_excel(base + "HorizontalActivities.xlsx")
JST_DG      = pd.read_excel(base + "SSM_DG.xlsx")

# ── Split SSPs and RTF ───────────────────────────────────────────────────────
SSP_obj = RTF_Extract[RTF_Extract["data_name"].str.startswith("SSP_Objective")].copy()

# Add DG to RTF
RTF = RTF_Extract.merge(JST_DG, on="jst_code", how="left").rename(columns={"ssm_dg": "DG"})

# ── Filter RTF to legend entries ─────────────────────────────────────────────
legend = [
    "SSP_RTF_CRE_Other_tolerance", "SSP_RTF_CRE_tolerance1_final",
    "SSP_RTF_E1_BM_Other_tolerance", "SSP_RTF_E1_BM_tolerance1_final",
    "SSP_RTF_E2_IGRM_tolerance1_final", "SSP_RTF_E3_B1_CCR_Other_tolerance",
    "SSP_RTF_E3_B1_CCR_tolerance1_final", "SSP_RTF_E3_B1_IRRBB_Other_tolerance",
    "SSP_RTF_E3_B1_MR_Other_tolerance", "SSP_RTF_E3_B1_OR_Other_tolerance",
    "SSP_RTF_E3_B1_OR_tolerance1_final", "SSP_RTF_E3_CA_Other_tolerance",
    "SSP_RTF_E4_B1_LR_Other_tolerance", "SSP_RTF_E4_B1_LR_tolerance1_final",
    "SSP_RTF_CRE_relevance", "SSP_RTF_E1_BM_relevance",
    "SSP_RTF_E2_IGRM_relevance", "SSP_RTF_E3_B1_CCR_relevance",
    "SSP_RTF_E3_B1_IRRBB_relevance", "SSP_RTF_E3_B1_MR_relevance",
    "SSP_RTF_E3_B1_OR_relevance", "SSP_RTF_E3_CA_relevance",
    "SSP_RTF_E4_B1_LR_relevance", "SSP_RTF_E2_IGRM_Other_tolerance",
    "SSP_RTF_E2_IGRM_tolerance2_final",
]

RTF = RTF[RTF["data_name"].isin(legend)].copy()
RTF["prior_key"] = RTF["srep_cycle"].astype(str) + "_" + RTF["data_name"]

# Join risk areas
RTF = RTF.merge(risk_areas, on="prior_key", how="left")
RTF = RTF.rename(columns={"Risk area": "risk_area"})
RTF = RTF.drop(columns=["prior_key", "Year", "data_name_y"], errors="ignore")
if "data_name_x" in RTF.columns:
    RTF = RTF.rename(columns={"data_name_x": "data_name"})

RTF["Key"] = RTF["srep_cycle"].astype(str) + "_" + RTF["jst_code"] + "_" + RTF["risk_area"]

# ── Primary risk mapping ─────────────────────────────────────────────────────
primary_risk_map = {
    "Gov_mangement_bodies":             "IGRM",
    "BM_Digital_transformation_strategies": "Business_model",
    "ClimateR_Material_exposure":       "Climate_risk",
    "CR_management_framework":          "Credit_risk",
    "ALM":                              "Liquidity_risk",
    "ClimateR_Management_of_CE_risks":  "Climate_risk",
    "Gov_RDARR":                        "IGRM",
    "OpR_Operational_resilience_framework": "Operational_risk",
    "LF_Funding_plans":                 "Liquidity_risk",
}
RTF["Primary_risk"] = RTF["risk_area"].replace(primary_risk_map)

# ── SSP objectives ───────────────────────────────────────────────────────────
ssp_conversion = {
    "CCR": "Credit_risk", "BM": "Business_model",
    "OR": "Operational_risk", "CA": "Capital_adequacy",
    "MR": "Market_risk", "LR": "Liquidity_risk",
    "Gov – Management bodies' functioning":               "Gov_mangement_bodies",
    "Business_model – Digital transformation strategies": "BM_Digital_transformation_strategies",
    "ClimateR – Physical and transition risk drivers":    "ClimateR_Management_of_CE_risks",
    "CR – CR management framework":                       "CR_management_framework",
    "ALM – Shortcomings in ALM framework":                "ALM",
    "ClimateR – Business strategies and management of C&E risks": "ClimateR_Management_of_CE_risks",
    "Gov – Risk data aggregation & reporting":            "Gov_RDARR",
    "OpR – Operational resilience framework":             "OpR_Operational_resilience_framework",
    "L&F – Funding sources and funding plans":            "LF_Funding_plans",
}
SSP_obj["value_txt"] = SSP_obj["value_txt"].replace(ssp_conversion)
SSP_obj["Key"] = (SSP_obj["srep_cycle"].astype(str) + "_"
                  + SSP_obj["jst_code"] + "_" + SSP_obj["value_txt"])

RTF["is_obj"] = RTF["Key"].isin(SSP_obj["Key"]).map({True: "Yes", False: "No"})

# ── OSIs ─────────────────────────────────────────────────────────────────────
OSIs["jst_code"] = OSIs["ID"].str[9:14]

mask = OSIs["Strategic objective"] == "No connection with SSM strategic objectives"
OSIs.loc[mask, "Strategic objective"] = OSIs.loc[mask, "Primary Risk Category"]

osi_conversion = {
    "CRED": "Credit_risk", "PROF": "Business_model",
    "OPER": "Operational_risk", "CAP": "Capital_adequacy",
    "IRRBB": "IRRBB", "GOV": "IGRM", "LIQ": "Liquidity_risk", "MARK": "Market_risk",
    "CR - CR management framework":                        "CR_management_framework",
    "ClimateR - Physical and transition risk drivers":     "ClimateR_Management_of_CE_risks",
    "L&F - Funding sources and funding plans":             "LF_Funding_plans",
    "Gov - Risk data aggregation & reporting":             "Gov_RDARR",
    "Gov - Management bodies' functioning":                "Gov_mangement_bodies",
    "ALM - Shortcomings in ALM frameworks":                "ALM",
    "BM - Digital transformation strategies":              "BM_Digital_transformation_strategies",
    "OpR - Operational resilience framework":              "OpR_Operational_resilience_framework",
    "ClimateR - Business strategies and management of C&E risks": "ClimateR_Management_of_CE_risks",
}
OSIs["Strategic objective"] = OSIs["Strategic objective"].replace(osi_conversion)
OSIs["Key"] = (OSIs["Launching Year"].astype(str) + "_"
               + OSIs["jst_code"] + "_" + OSIs["Strategic objective"])

osi_counts = OSIs.groupby("Key").size().rename("N_OSIs")
RTF = RTF.merge(osi_counts, on="Key", how="left")
RTF["N_OSIs"] = RTF["N_OSIs"].fillna(0).astype(int)

topic_osis = OSIs.groupby("Key")["Label"].apply(lambda x: ", ".join(x)).rename("Topic")
RTF = RTF.merge(topic_osis, on="Key", how="left")

# ── Horizontal activities ─────────────────────────────────────────────────────
horact_conversion = {
    "ClimateR - Physical and transition risk drivers":     "ClimateR_Management_of_CE_risks",
    "CR - CR management framework":                        "CR_management_framework",
    "BM - Digital transformation strategies":              "BM_Digital_transformation_strategies",
    "Gov - Management bodies' functioning":                "Gov_mangement_bodies",
    "OpR - Operational resilience framework":              "OpR_Operational_resilience_framework",
    "L&F - Funding sources and funding plans":             "LF_Funding_plans",
    "ALM - Shortcomings in ALM frameworks":                "ALM",
    "Gov - Risk data aggregation & reporting":             "Gov_RDARR",
}
HorAct["Strategic objective"] = HorAct["Strategic objective"].replace(horact_conversion)
HorAct["Key"] = (HorAct["Planning year"].astype(str) + "_"
                 + HorAct["JST code"] + "_" + HorAct["Strategic objective"])

horact_counts = HorAct.groupby("Key").size().rename("N_Horizontal")
RTF = RTF.merge(horact_counts, on="Key", how="left")
RTF["N_Horizontal"] = RTF["N_Horizontal"].fillna(0).astype(int)

# ── SREP scores ───────────────────────────────────────────────────────────────
MYA_info = pd.read_excel(base + "MYA_Assessment.xlsx")

MYA_info.loc[MYA_info["element_shortname"].str.startswith("CAP", na=False), "element_risk"] = "CAP"

srep_risk_map = {
    "CR": "Credit_risk", "BM": "Business_model", "OR": "Operational_risk",
    "CAP": "Capital_adequacy", "MR": "Market_risk", "LR": "Liquidity_risk",
}
MYA_info["element_risk"] = MYA_info["element_risk"].replace(srep_risk_map)
MYA_info["Key"] = (MYA_info["srep_cycle"].astype(str) + "_"
                   + MYA_info["jst_code"] + "_" + MYA_info["element_risk"])

combined_mask = (
    MYA_info["element_shortname"].str.endswith("Combined score", na=False)
    | MYA_info["element_shortname"].str.endswith("Combined Score", na=False)
    | MYA_info["element_shortname"].str.endswith("CAP – Overall ICAAP assessment", na=False)
) & MYA_info["engagement_name"].str.startswith("SREP", na=False)

SREP_Scores = MYA_info[combined_mask][["jst_code", "srep_cycle", "element_risk", "Key", "latest_score"]]
RTF = RTF.merge(SREP_Scores[["Key", "latest_score"]], on="Key", how="left")

overall_mask = (
    (MYA_info["element_fullname"] == "Overall SREP score assessment")
    & MYA_info["engagement_name"].str.startswith("SREP", na=False)
)
overall_SREP = MYA_info[overall_mask][["Key", "element_fullname", "jst_code", "srep_cycle", "latest_score"]]
RTF = RTF.merge(overall_SREP, on=["jst_code", "srep_cycle"], how="left")

# ── MYA update status ─────────────────────────────────────────────────────────
mya_filtered = MYA_info[~MYA_info["element_shortname"].str.endswith("Combined score", na=False)]

def agg_modules(group, condition, col="element_shortname"):
    filtered = group[condition(group)]
    return pd.Series({
        "count": len(filtered),
        "names": ", ".join(filtered[col]),
    })

update_status = mya_filtered.groupby("Key").apply(lambda g: pd.Series({
    "N_of_modules_updated":          (g["scopeofassessmenttype"] == "Updated").sum(),
    "UpdatedModules":                ", ".join(g.loc[g["scopeofassessmenttype"] == "Updated", "element_shortname"]),
    "N_of_notupdated_nonmaterial":   (g["assessmentreason"] == "Non-material module for the bank").sum(),
    "NotUpdated_nonmaterial":        ", ".join(g.loc[g["assessmentreason"] == "Non-material module for the bank", "element_shortname"]),
    "N_of_notupdated_RTF":           (g["assessmentreason"] == "Updated assessment in next SREP cycles in line with the RTF").sum(),
    "NotUpdated_RTF":                ", ".join(g.loc[g["assessmentreason"] == "Updated assessment in next SREP cycles in line with the RTF", "element_shortname"]),
    "N_of_notupdated_otheractivity": (g["assessmentreason"] == "Upcoming/ongoing non-SREP activity with outcome for next SREP cycle").sum(),
    "NotUpdated_otheractivity":      ", ".join(g.loc[g["assessmentreason"] == "Upcoming/ongoing non-SREP activity with outcome for next SREP cycle", "element_shortname"]),
    "N_of_notupdated_remedial":      (g["assessmentreason"] == "Supervisory activity from previous cycle with ongoing remedial actions").sum(),
    "NotUpdated_remedial":           ", ".join(g.loc[g["assessmentreason"] == "Supervisory activity from previous cycle with ongoing remedial actions", "element_shortname"]),
    "N_of_notupdated_other":         ((g["assessmentreason"] == "not defined reason") & (g["element_module_name"] != "N/A")).sum(),
    "NotUpdated_others":             ", ".join(g.loc[(g["assessmentreason"] == "not defined reason") & (g["element_module_name"] != "N/A"), "element_shortname"]),
})).reset_index()

RTF = RTF.merge(update_status, on="Key", how="left")

int_cols = [
    "N_of_modules_updated", "N_of_notupdated_nonmaterial", "N_of_notupdated_RTF",
    "N_of_notupdated_otheractivity", "N_of_notupdated_remedial", "N_of_notupdated_other",
]
str_cols = [
    "UpdatedModules", "NotUpdated_nonmaterial", "NotUpdated_RTF",
    "NotUpdated_otheractivity", "NotUpdated_remedial", "NotUpdated_others",
]
RTF[int_cols] = RTF[int_cols].fillna(0).astype(int)
RTF[str_cols] = RTF[str_cols].fillna("")

RTF["TotalModules"] = RTF[int_cols].sum(axis=1)
RTF["Updated_Percentage"] = (
    (RTF["N_of_modules_updated"] / RTF["TotalModules"].replace(0, pd.NA) * 100)
    .round(0)
    .astype("Int64")
    .astype(str) + "%"
)

# ── Readiness / FTEs ──────────────────────────────────────────────────────────
Readiness = pd.read_excel(
    "P:/ECB business areas/SSM-DSSR/Strategic and Planning office/"
    "PBI extraction files [Do not delete or move]/FTEs.xlsx",
    sheet_name="Sheet1"
)

readiness_map = {
    "Credit Risk": "Credit_risk", "Business Model": "Business_model",
    "Operational Risk": "Operational_risk", "Capital Adequacy": "Capital_adequacy",
    "Market Risk": "Market_risk", "Governance Risk": "IGRM",
    "IT and Cyber Risk": "OpR_Operational_resilience_framework",
    "Liquidity Risk": "Liquidity_risk",
}
Readiness["risk_area"] = Readiness["risk_area"].replace(readiness_map)
Readiness = Readiness[Readiness["risk_area"] != "Operational, IT and Cyber Risk"]
Readiness = Readiness[Readiness["name"] != "BAWAG Group AG (should be MTHSB)"]
Readiness["srep_cycle"] = "2024"
Readiness["Key"] = (Readiness["srep_cycle"].astype(str) + "_"
                    + Readiness["jst_code"] + "_" + Readiness["risk_area"])

RTF = RTF.merge(Readiness, on="Key", how="left", suffixes=("", "_r"))
drop_cols = [c for c in RTF.columns if c.endswith("_r")]
RTF = RTF.drop(columns=drop_cols)

# ── Export ────────────────────────────────────────────────────────────────────
RTF.to_csv("RTF_objectives_trial.csv", index=False)
print("Done — saved to RTF_objectives_trial.csv")