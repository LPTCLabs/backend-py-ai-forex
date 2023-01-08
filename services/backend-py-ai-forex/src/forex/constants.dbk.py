from pathlib import Path

BASE_PATH = Path(__file__).parent

DATA_PATH: str = f"{BASE_PATH}/data"
MACRO_DATA: str = f"{DATA_PATH}/macro_data-all_data_2.0.csv"
MACRO_DELAY_SOURCE: str = f"{DATA_PATH}/delay_macro.csv"
TRAIN_DEV_DATA: str = f"{DATA_PATH}/ML_Data_29un2022-train+dev.csv"
TRAIN_DATA: str = f"{DATA_PATH}/ML_Data_29un2022-train.csv"
DEV_DATA: str = f"{DATA_PATH}/ML_Data_29un2022-train.csv"
TEST_DATA: str = f"{DATA_PATH}/ML_Data_29un2022-test.csv"

NB_MACRO_DATA: int = 15  # 15, 188
TRAINING_WITH_TRAIN_DEV_DATA: bool = True
MODEL_TYPE = "logistic"  # logistic, SVM, random_forest, decision_tree, xgboost
ADD_MOVING_AVERAGE: bool = True
ADD_MACRO: bool = True
ADD_COMMODITY: bool = True
ADD_FI: bool = True
ADD_EQUITY: bool = True
WINDOW_PROBABILITY_ANALYSIS_LONGEST: int = 6
WINDOW_PROBABILITY_ANALYSIS_NUMBER: int = 3
PLOT_VP_SIX_MONTHS: bool = True
SMA: bool = False
SMA_WINDOW: int = 20
SMA_PLOT: bool = True

# BINARY CLASSIFICATION
BINARY_LABELS: list = ["down", "up"]
# BINARY_THRESHOLD_FOREX: float = 0.12
BINARY_THRESHOLD_FOREX: float = 0.117
US_EA_BINARY: dict = {
    "MACRO": [
        "FI_EA_IR3TIB01EZM156N",
        # "EA_Wage_Growth_YoY",  # accuracy drops extremely with this feature
        # "EA_Business_Confidence",
        # "EA_Business_Confidence2",
        "EA_Composite_PMI_Final",
        # "EA_Consumer_Confidence",
        # "EA_Consumer_Confidence_Final",  # accuracy drops a lot  with this feature
        # "EA_Consumer_Confidence_Price_Trends",  # accuracy drops a lot with this feature
        # "EA_Consumer_Inflation_Expectations",
        # "EA_Economic_Optimism_Index",  # accuracy drops a lot with this feature
        "EA_Global_Construction_PMI",
        # "EA_Industrial_Sentiment",  # accuracy drops a lot with this feature
        "EA_Industrial_Sentiment2",
        # "EA_Manufacturing_PMI_Final",
        "EA_Mining_Production",
        "EA_PPI_MoM",
        # "EA_Producer_Prices",  # accuracy drops a lot with this feature
        # "EA_Producer_Price_Change",
        "EA_Retail_Sales_MoM",
        # "EA_Services_PMI_Final",  # accuracy drops extremely with this feature
        # "EA_Services_Sentiment",  # accuracy drops extremely with this feature
        # "EA_Unemployment_Rate",
        # "FR_Business_Climate_Indicator",  # accuracy drops extremely with this feature
        # "FR_Business_Confidence",  # accuracy drops extremely with this feature
        "FR_Composite_PMI_Final",
        # "FR_Consumer_Confidence",
        "FR_GDP_Growth_Rate_QoQ_Final",
        "FR_Global_Construction_PMI",
        "FR_Manufacturing_PMI_Final",
        "FR_PPI_MoM",
        # "FR_Retail_Sales_MoM",
        "FR_Services_PMI",
        "FR_Unemployment_Rate",
        # "GR_Business_Confidence",  # accuracy drops a lot with this feature
        "GR_Consumer_Confidence",
        "GR_Manufacturing_PMI",
        # "GR_Unemployment_Rate",
        # "IE_Consumer_Confidence",
        "IE_GDP_Growth_Rate_QoQ",
        "IE_GDP_Growth_Rate_QoQ_Final",
        # "IE_Retail_Sales_MoM",
        # "IE_Unemployment_Rate",
        "IE_Wholesale_Prices_MoM",
        # "IT_Business_Confidence",  # accuracy drops extremely with this feature
        # "IT_Consumer_Confidence",  # accuracy drops extremely with this feature
        # "IT_GDP_Growth_Rate_QoQ_Final",
        "IT_Global_Construction_PMI",
        "IT_PPI_MoM",
        "IT_Retail_Sales_MoM",
        "IT_Unemployment_Rate",
        # "ES_Consumer_Confidence",  # accuracy drops extremely with this feature
        # "ES_GDP_Growth_Rate_QoQ_Final",
        "ES_Manufacturing_PMI",
        # "ES_Retail_Sales_MoM",
        "ES_Services_PMI",
        "ES_Unemployment_Change",  # makes recall down better
        "ES_Unemployment_Rate",
        "DE_Composite_PMI_Final",
        # "DE_GDP_Growth_Rate_QoQ_Final",
        "DE_Global_Construction_PMI",
        # "DE_Import_Prices_MoM",
        # "DE_Manufacturing_PMI_Final",  # accuracy drops extremely with this feature
        # "DE_PPI_MoM",
        # "DE_Retail_Sales_MoM",
        # "DE_Services_PMI_Final",
        # "DE_Unemployment_Rate",
        "DE_Wholesale_Prices_MoM",
        # "BE_Business_Confidence",
        # "BE_Consumer_Confidence",
        # "BE_GDP_Growth_Rate_QoQ_Final",
        # "BE_Retail_Sales_MoM",
        # "AT_Business_Confidence",
        # "AT_Consumer_Confidence",
        # "AT_GDP_Growth_Rate_QoQ_Final",
        # "AT_PPI_MoM",
        # "AT_Retail_Sales_MoM",
        "AT_Wholesale_Prices_MoM",
        "PT_Business_Confidence",
        # "PT_Consumer_Confidence",
        # "PT_GDP_Growth_Rate_QoQ_Final",
        "PT_PPI_MoM",
        # "PT_Retail_Sales_MoM",
        "PT_Unemployment_Rate",
        "NL_Business_Confidence",
        # "NL_Consumer_Confidence",
        # "NL_GDP_Growth_Rate_QoQ_Final",  # accuracy drops a lot with this feature
        # "NL_Unemployment_Rate",
        # "AHETPI",
        # "BOPGEXP",
        # "BOPGIMP",  # accuracy drops extremely with this feature
        # "BOPGSTB",
        # "BOPGTB",  # accuracy drops extremely with this feature
        # "BOPSEXP",
        # "BOPSIMP",
        # "BOPSTB",
        # "BOPTEXP",
        # "BOPTIMP",  # accuracy drops extremely with this feature
        "CEFDFSA066MSFRBPHI",
        "CES0800000001",  # accuracy drops a lot without this feature
        # "CES4244800001",
        "CDSP",
        # "CPIAUCSL",
        # "CSUSHPISA",
        "CURRDD",  # recall down is better with this feature
        "CURRSL",  # accuracy drops extremely without this feature
        "CWSR0000SA0",  # accuracy drops a lot without this feature
        # "DEMDEPSL",  # accuracy drops a lot with this feature
        # "DGDSRC1",
        # "DMANEMP",
        "DSPI",  # accuracy drops extremely without this feature
        "FEDFUNDS",
        "FODSP",
        "FRBKCLMCILA",
        # "FRBKCLMCIM",  # accuracy drops a lot with this feature
        "GACDFSA066MSFRBPHI",
        # "GACDISA066MSFRBNY",
        # "HPIPONM226S",
        # "INDPRO",
        "IPCONGD",
        "IPMANSICS",  # accuracy drops a lot without this feature
        # "IPMINE",  # accuracy drops a lot with this feature
        "IPUTIL",
        "IR3TIB01USM156N",
        # "ITMTAEM133S",  # accuracy drops extremely with this feature
        # "ITXFISM133S",  # accuracy drops a lot with this feature
        # "ITXTCIM133S",  # accuracy drops extremely with this feature
        # "JTS1000JOL",
        # "JTS2300JOL",  # accuracy drops extremely with this feature
        # "JTS3000JOL",
        # "JTS4000JOL",
        # "JTS4400JOL",
        # "JTS540099JOL", # accuracy drops extremely with this feature
        # "JTS6000JOL",  # accuracy drops extremely with this feature
        # "JTS7000JOL",  # accuracy drops extremely with this feature
        # "JTS7200JOL",  # accuracy drops extremely with this feature
        # "JTSJOL",
        # "JTSOSL",
        "M1REAL",
        "M1SL",
        "M2REAL",  # accuracy drops a lot without this feature
        "M2SL",
        "MANEMP",  # accuracy drops a lot without this feature
        "MCUMFN",
        # "MEIM683SFRBCHI",
        # "MNFCTRIMSA",
        "MNFCTRIRSA",
        # "MNFCTRMPCIMSA",  # makes recall down worse
        # "MNFCTRMPCSMSA",
        # "MNFCTRSMSA",  # accuracy drops a lot with this feature
        "MPCT00XXS",
        # "MPCT03XXS",
        # "MPCT04XXS",
        # "MPCT12XXS",
        # "MPCTNRXXS",
        # "MPCTXXXXS",
        "MPCV00XXS",
        # "NPPTTL",
        # "PCE",
        # "PCEC96",  # accuracy drops extremely with this feature
        "PCEDG",  # accuracy drops a lot without this feature
        # "PCEDGC96",  # accuracy drops extremely with this feature
        "PCEND",
        # "PCENDC96",  # accuracy drops extremely with this feature
        "PCEPI",
        "PCEPILFE",
        "PCES",
        # "PCESC96",  # accuracy drops extremely with this feature
        "PCTR",
        # "PDI",  # accuracy drops extremely with this feature
        # "PI",
        # "PMSAVE",
        "PPCDFSA066MSFRBPHI",
        # "PPCDISA066MSFRBNY",
        # "PRRESCONS",  # accuracy drops a lot with this feature
        "PSAVERT",
        "RMFSL",
        "RPI",
        # "S4248SM144SCEN",
        # "SPCS20RSA",
        "STDSL",
        "TCU",
        # "TEMPHELPS",
        # "TLCOMCONS",  # accuracy drops a lot with this feature
        # "TLHLTHCONS",  # accuracy drops a lot with this feature
        # "TLHWYCONS",  # accuracy drops a lot with this feature
        # "TLNRESCONS",
        # "TLRESCONS",  # accuracy drops extremely with this feature
        # "TTLCONS",  # accuracy drops a lot with this feature
        "USCONS",
        "USFIRE",
        "USGOOD",
        "USPBS",  # accuracy drops a lot without this feature
        # "USPHCI",
        "USPRIV",
        # "USSLIND",
        "USTPU",
        "USTRADE",  # accuracy drops a lot without this feature
        "W825RC1",  # accuracy drops a lot without this feature
        "W875RX1",
        "WHLSLRIRSA",
        "A939RX0Q048SBEA",
        "B009RC1Q027SBEA",  # accuracy drops a lot without this feature
        # "COFC",
        "CP",
        "CPATAX",  # accuracy drops a lot without this feature
        "FPI",  # accuracy drops extremely without this feature
        "GDI",
        "GDPC1",
        "GGSAVE",
        # "GPDI",
        "GPDIC1",
        "GPSAVE",
        # "GSAVE",
        # "IEABC",
        # "IEAMGS",
        # "IEAXGS",  # accuracy drops extremely with this feature
        # "MDSP",
        # "MMMFTAQ027S",
        "NETFI",  # accuracy drops a lot without this feature
        # "OUTMS",
        "PNFI",  # accuracy drops a lot without this feature
        "PRFI",
        "RSAHORUSQ156S",
        # "TDSP",
        # "ULCMFG",
        # "W207RC1Q156SBEA",
        "W986RC1Q027SBEA",  # accuracy drops a lot without this feature
        # "IPG211111CS",  # accuracy drops a lot with this feature
        # "IPG3361T3S",  # accuracy drops a lot with this feature
        # "IPMAN",  # accuracy drops a lot with this feature
        # "IPN3311A2RS",  # accuracy drops a lot with this feature
        # "CAPUTLG3311A2S",
        # "CES1021100001",
        # "CES4244110001",
        "IPN31152S",
        # "CIVPART",
        # "USALOLITONOSTSAM",
        # "MSACSR",
        # "UNRATE",
        # "SAHMREALTIME",
        # "CORESTICKM159SFRBATL",
        # "PAYEMS",
        # "TOTALSA",
        "RETAILIRSA",
        # "HOUST",
        # "CE16OV",
        "ISRATIO",
        # "DSPIC96",
        # "MABMM301USM189S",
        # "ALTSALES",
        # "DAUPSA",
        "RETAILIMSA",
        # "RSAFS",
        "CLF16OV",
        # "AISRSA",
        # "TOTALSL",
        # "MYAGM2USM052S",
        # "LNS14000006",
        # "CUSR0000SETA02",
        # "SAHMCURRENT",
        "PERMIT",
        "MANEMP",
        # "USALORSGPNOSTSAM",
        # "STICKCPIM157SFRBATL",
        # "CPILFESL",
        # "BUSINV",
        "RRSFS",
        # "CPIUFDSL",
        # "BUSLOANS",  # makes recall down worse
        # "JTSLDL",  # accuracy drops extremely with this feature
        # "REVOLSL",
        # "VMTD11",
        # "A229RX0",  ######## makes recall down worse but improve the accuracy
        # "CUSR0000SAH1",
        "HSN1F",  # improve the accuracy without making recall down worse
        # "LNS11300060",
        # "RSCCAS",  # accuracy drops a lot with this feature
        # "PCETRIM12M159SFRBDAL",
        # "LFWA64TTUSM647S",
        "LNS11300002",
        "HTRUCKSSAAR",  ########
        # "EMRATIO",
        # "TSIFRGHTC",
        # "MRTSSM44X72USS",
        "CAUR",
        "CMRMTSPL",
        # "USAUCSFRCONDOSMSAMID",
        # "FLUR",
        # "UNDCONTSA",  # accuracy drops a lot with this feature
        # "CUSR0000SEHA",
        # "LMJVTTUVUSM647S",
        # "RAILFRTCARLOADSD11",  # accuracy drops extremely without this feature
        # "USGOVT",
    ],
    "COMMODITY": [
        # "Aluminum",  # accuracy drops a lot with this feature
        # "Baltic Dry",  # accuracy drops a lot with this feature
        "Brent",
        # "Coal",  # accuracy drops extremely with this feature
        "Cocoa",  # accuracy drops extremely without this feature
        # "Coffee",  # accuracy drops a lot with this feature
        "Copper",
        # "Corn",  # accuracy drops extremely with this feature
        # "Cotton",
        # "Crude Oil",
        "Ethanol",
        "Feeder Cattle",
        "Gasoline",
        # "Gold",  # accuracy drops a lot with this feature
        "Heating Oil",
        "Iron Ore",
        # "Lead",  # accuracy drops a lot with this feature
        # "Lean Hogs",
        # "Live Cattle",
        # "LME Index",  # accuracy drops extremely with this feature
        "Lumber",
        "Methanol",
        "Natural gas",
        # "Nickel",  # accuracy drops extremely with this feature
        "Propane",
        # "S&P GSCI",  # accuracy drops extremely with this feature
        # "Silver",
        # "Soybeans",  # accuracy drops a lot with this feature
        # "Sugar",  # accuracy drops extremely with this feature
        # "Tin",  # accuracy drops a lot with this feature
        # "Wheat",  # accuracy drops extremely with this feature
        # "Zinc",  # accuracy drops extremely with this feature
    ],
    "FI": [
        "FI_T10YFFM",
        "FI_TB3SMFFM",
        "FI_GS10",
        "FI_TB3MS",
        "FI_EA_IRLTLT01EZM156N",
        "FI_T10Y2YM",
        # "FI_GS2",
        # "FI_T10YIEM",
        # "FI_T5YIEM",  # accuracy drops a lot with this feature
        # "FI_GS30",
    ],
    "EQUITY": [
        "EQ_EA_DAX30",  # accuracy drops a lot with this feature
        "EQ_EA_CAC40",
        "EQ_EA_AEX",  # accuracy drops a lot with this feature
        "EQ_EA_IBEX",
        "EQ_Dow_Jones",
        # "EQ_NASDAQ",  # accuracy drops a lot with this feature
        # "EQ_SPX",   # accuracy drops a lot with this feature
    ],
}


# Outliers
US_EA_BINARY: dict = {
    "MACRO": [
        "FI_EA_IR3TIB01EZM156N",
        # "EA_Wage_Growth_YoY",  # accuracy drops extremely with this feature and many outliers
        # "EA_Business_Confidence",  # many outliers
        # "EA_Business_Confidence2",
        # "EA_Composite_PMI_Final",  # many outliers
        # "EA_Consumer_Confidence",  # 6 outliers
        # "EA_Consumer_Confidence_Final",  # accuracy drops a lot  with this feature and many outliers
        # "EA_Consumer_Confidence_Price_Trends",  # accuracy drops a lot with this feature and many outliers
        # "EA_Consumer_Inflation_Expectations",  # no outliers
        # "EA_Economic_Optimism_Index",  # accuracy drops a lot with this feature and many outliers
        "EA_Global_Construction_PMI",  # no outliers
        # "EA_Industrial_Sentiment",  # accuracy drops a lot with this feature and many outliers and many outliers
        "EA_Industrial_Sentiment2",  # many outliers
        # "EA_Manufacturing_PMI_Final",  # many outliers
        "EA_Mining_Production",  # many outliers
        "EA_PPI_MoM",  # many outliers
        # "EA_Producer_Prices",  # accuracy drops a lot with this feature and many outliers
        # "EA_Producer_Price_Change",  # many outliers
        "EA_Retail_Sales_MoM",  # many outliers
        # "EA_Services_PMI_Final",  # accuracy drops extremely with this feature and many outliers
        # "EA_Services_Sentiment",  # accuracy drops extremely with this feature and many outliers
        # "EA_Unemployment_Rate",  # 2 outliers
        # "FR_Business_Climate_Indicator",  # accuracy drops extremely with this feature
        # "FR_Business_Confidence",  # accuracy drops extremely with this feature and no outliers
        "FR_Composite_PMI_Final",  # many outliers
        # "FR_Consumer_Confidence",  # many outliers
        "FR_GDP_Growth_Rate_QoQ_Final",  # 4 outliers
        "FR_Global_Construction_PMI",  # no outliers
        "FR_Manufacturing_PMI_Final",  # many outliers
        "FR_PPI_MoM",  # many outliers
        # "FR_Retail_Sales_MoM",  # 6 outliers
        "FR_Services_PMI",
        "FR_Unemployment_Rate",  # 5 outliers
        # "GR_Business_Confidence",  # accuracy drops a lot with this feature and no outliers
        "GR_Consumer_Confidence",  # no outliers
        "GR_Manufacturing_PMI",  # many outliers
        # "GR_Unemployment_Rate",  # no outliers
        # "IE_Consumer_Confidence",  # no outliers
        "IE_GDP_Growth_Rate_QoQ",  # no outliers
        "IE_GDP_Growth_Rate_QoQ_Final",  # 1 outlier (bad)
        # "IE_Retail_Sales_MoM",  # many outliers
        # "IE_Unemployment_Rate",  # no outliers
        "IE_Wholesale_Prices_MoM",  # 8 outliers
        # "IT_Business_Confidence",  # accuracy drops extremely with this feature and many outliers
        # "IT_Consumer_Confidence",  # accuracy drops extremely with this feature and no outliers
        # "IT_GDP_Growth_Rate_QoQ_Final",  # 9 outliers
        "IT_Global_Construction_PMI",  # many outliers
        "IT_PPI_MoM",  # many outliers
        "IT_Retail_Sales_MoM",  # many outliers
        "IT_Unemployment_Rate",  # no outliers
        # "ES_Consumer_Confidence",  # accuracy drops extremely with this feature and no outliers
        # "ES_GDP_Growth_Rate_QoQ_Final",  # 6 outliers
        "ES_Manufacturing_PMI",  # many outliers
        # "ES_Retail_Sales_MoM",  # many outliers
        "ES_Services_PMI",  # many outliers
        "ES_Unemployment_Change",  # makes recall down better and many outliers
        "ES_Unemployment_Rate",  # no outliers
        "DE_Composite_PMI_Final",  # many outliers (bad)
        # "DE_GDP_Growth_Rate_QoQ_Final", # 8 outliers
        "DE_Global_Construction_PMI",  # 1 outlier
        # "DE_Import_Prices_MoM",  # many outliers
        # "DE_Manufacturing_PMI_Final",  # accuracy drops extremely with this feature and many outliers
        # "DE_PPI_MoM",  # many outliers
        # "DE_Retail_Sales_MoM",  # many outliers
        # "DE_Services_PMI_Final",  # many outliers
        # "DE_Unemployment_Rate",  # 7 outliers
        "DE_Wholesale_Prices_MoM",  # many outliers
        # "BE_Business_Confidence",  # 8 outliers
        # "BE_Consumer_Confidence",  # 9 outliers
        # "BE_GDP_Growth_Rate_QoQ_Final",  # 8 outliers
        # "BE_Retail_Sales_MoM",  # many outliers
        # "AT_Business_Confidence",  # many outliers
        # "AT_Consumer_Confidence",  # many outliers
        # "AT_GDP_Growth_Rate_QoQ_Final",  # 3 outliers
        # "AT_PPI_MoM",  # many outliers
        # "AT_Retail_Sales_MoM",  # many outliers
        "AT_Wholesale_Prices_MoM",  # many outliers
        "PT_Business_Confidence",  # 2 outliers
        # "PT_Consumer_Confidence",  # many outliers
        # "PT_GDP_Growth_Rate_QoQ_Final",  # many outliers
        "PT_PPI_MoM",  # many outliers
        # "PT_Retail_Sales_MoM",  # many outliers
        "PT_Unemployment_Rate",  # no outliers
        "NL_Business_Confidence",  # many outliers
        # "NL_Consumer_Confidence",  # 2 outliers
        # "NL_GDP_Growth_Rate_QoQ_Final",  # accuracy drops a lot with this feature and 4 outliers
        # "NL_Unemployment_Rate",  # no outliers
        # "AHETPI",  # no outliers
        # "BOPGEXP",  # no outliers
        # "BOPGIMP",  # accuracy drops extremely with this feature and no outliers
        # "BOPGSTB",  # 1 outlier
        # "BOPGTB",  # accuracy drops extremely with this feature and no outliers
        # "BOPSEXP",  # no outliers
        # "BOPSIMP",  # no outliers
        # "BOPSTB",  # no outliers
        # "BOPTEXP",  # no outliers
        # "BOPTIMP",  # accuracy drops extremely with this feature and no outliers
        "CEFDFSA066MSFRBPHI",  # many outliers
        "CES0800000001",  # accuracy drops a lot without this feature and no outliers
        # "CES4244800001",  # many outliers
        "CDSP",  # no outliers
        # "CPIAUCSL",  # no outliers
        # "CSUSHPISA",  # no outliers
        "CURRDD",  # recall down is better with this feature and many outliers
        "CURRSL",  # accuracy drops extremely without this feature and many outliers
        "CWSR0000SA0",  # accuracy drops a lot without this feature no outliers
        # "DEMDEPSL",  # accuracy drops a lot with this feature and many outliers
        # "DGDSRC1",  # no outliers
        # "DMANEMP",  # no outliers
        "DSPI",  # accuracy drops extremely without this feature and no outliers
        "FEDFUNDS",  # many outliers
        "FODSP",  # 1 outlier
        "FRBKCLMCILA",  # no outliers
        # "FRBKCLMCIM",  # accuracy drops a lot with this feature and many outliers
        "GACDFSA066MSFRBPHI",  # many outliers
        # "GACDISA066MSFRBNY",  # 4 outliers
        # "HPIPONM226S",  # no outliers
        # "INDPRO",  # no outliers
        "IPCONGD",  # no outliers
        "IPMANSICS",  # accuracy drops a lot without this feature and no outliers
        # "IPMINE",  # accuracy drops a lot with this feature and many outliers
        "IPUTIL",  # no outliers
        "IR3TIB01USM156N",  # many outliers
        # "ITMTAEM133S",  # accuracy drops extremely with this feature and 6 outliers
        # "ITXFISM133S",  # accuracy drops a lot with this feature and no outliers
        # "ITXTCIM133S",  # accuracy drops extremely with this feature and no outliers
        # "JTS1000JOL",  # 6 outliers
        # "JTS2300JOL",  # accuracy drops extremely with this feature and 6 outliers
        # "JTS3000JOL",  # many outliers
        # "JTS4000JOL",  # 5 outliers
        # "JTS4400JOL",  # 2 outliers
        # "JTS540099JOL", # accuracy drops extremely with this feature and 3 outliers
        # "JTS6000JOL",  # accuracy drops extremely with this feature and 3 outliers
        # "JTS7000JOL",  # accuracy drops extremely with this feature and 6 outliers
        # "JTS7200JOL",  # accuracy drops extremely with this feature and 7 outliers
        # "JTSJOL",  # 7 outliers
        # "JTSOSL",  # no outliers
        "M1REAL",  # many outliers (bad)
        "M1SL",  # many outliers (bad)
        "M2REAL",  # accuracy drops a lot without this feature and many outliers
        "M2SL",  # many outliers
        "MANEMP",  # accuracy drops a lot without this feature and no outliers
        "MCUMFN",  # many outliers
        # "MEIM683SFRBCHI",  # many outliers
        # "MNFCTRIMSA",  # many outliers
        "MNFCTRIRSA",  # 3 outliers
        # "MNFCTRMPCIMSA",  # makes recall down worse and 10 outliers
        # "MNFCTRMPCSMSA",  # many outliers
        # "MNFCTRSMSA",  # accuracy drops a lot with this feature and no outliers
        "MPCT00XXS",  # 5 outliers
        # "MPCT03XXS",  # 2 outliers
        # "MPCT04XXS",  # 2 outliers
        # "MPCT12XXS",  # many outliers
        # "MPCTNRXXS",  # 5 outliers
        # "MPCTXXXXS",  # 4 outliers
        "MPCV00XXS",  # 5 outliers
        # "NPPTTL",  # no outliers
        # "PCE",  # no outliers
        # "PCEC96",  # accuracy drops extremely with this feature and no outliers
        "PCEDG",  # accuracy drops a lot without this feature and no outliers
        # "PCEDGC96",  # accuracy drops extremely with this feature and 2 outliers
        "PCEND",  # no outliers
        # "PCENDC96",  # accuracy drops extremely with this feature and 7 outliers
        "PCEPI",  # no outliers
        "PCEPILFE",  # no outliers
        "PCES",  # no outliers
        # "PCESC96",  # accuracy drops extremely with this feature and no outliers
        "PCTR",  # 6 outliers
        # "PDI",  # accuracy drops extremely with this feature and no outliers
        # "PI",  # no outliers
        # "PMSAVE",  # many outliers (bad)
        "PPCDFSA066MSFRBPHI",  # 3 outliers
        # "PPCDISA066MSFRBNY",  # no outliers
        # "PRRESCONS",  # accuracy drops a lot with this feature and no outliers
        "PSAVERT",  # 7 outliers
        "RMFSL",
        "RPI",  # no outliers
        # "S4248SM144SCEN",  # no outliers
        # "SPCS20RSA",  # no outliers
        "STDSL",  # no outliers
        "TCU",  # 6 outliers
        # "TEMPHELPS",  # no outliers
        # "TLCOMCONS",  # accuracy drops a lot with this feature and no outliers
        # "TLHLTHCONS",  # accuracy drops a lot with this feature and many outliers
        # "TLHWYCONS",  # accuracy drops a lot with this feature and no outliers
        # "TLNRESCONS",  # no outliers
        # "TLRESCONS",  # accuracy drops extremely with this feature and no outliers
        # "TTLCONS",  # accuracy drops a lot with this feature and no outliers
        "USCONS",  # no outliers
        "USFIRE",  # no outliers
        "USGOOD",  # no outliers
        "USPBS",  # accuracy drops a lot without this feature and no outliers
        # "USPHCI",  # no outliers
        "USPRIV",  # no outliers
        # "USSLIND",  # many outliers (bad)
        "USTPU",  # no outliers
        "USTRADE",  # accuracy drops a lot without this feature and no outliers
        "W825RC1",  # accuracy drops a lot without this feature and many outliers (bad)
        "W875RX1",  # no outliers
        "WHLSLRIRSA",  # 2 outliers
        "A939RX0Q048SBEA",  # no outliers
        "B009RC1Q027SBEA",  # accuracy drops a lot without this feature and no outliers
        # "COFC",  # no outliers
        "CP",  # no outliers
        "CPATAX",  # accuracy drops a lot without this feature and no outliers
        "FPI",  # accuracy drops extremely without this feature and no outliers
        "GDI",  # no outliers
        "GDPC1",  # no outliers
        "GGSAVE",  # many outliers (bad)
        # "GPDI",  # no outliers
        "GPDIC1",  # no outliers
        "GPSAVE",  # 6 outliers
        # "GSAVE",  # no outliers
        # "IEABC",  # no outliers
        # "IEAMGS",  # no outliers
        # "IEAXGS",  # accuracy drops extremely with this feature and no outliers
        # "MDSP",   # no outliers
        # "MMMFTAQ027S",  # no outliers
        "NETFI",  # accuracy drops a lot without this feature and no outliers
        # "OUTMS",  # no outliers
        "PNFI",  # accuracy drops a lot without this feature and no outliers
        "PRFI",   # no outliers
        "RSAHORUSQ156S",  # no outliers
        # "TDSP",  # no outliers
        # "ULCMFG",  # many outliers
        # "W207RC1Q156SBEA", # 2 outliers
        "W986RC1Q027SBEA",  # accuracy drops a lot without this feature and many outliers (bad)
        # "IPG211111CS",  # accuracy drops a lot with this feature and no outliers
        # "IPG3361T3S",  # accuracy drops a lot with this feature and no outliers
        # "IPMAN",  # accuracy drops a lot with this feature and no outliers
        # "IPN3311A2RS",  # accuracy drops a lot with this feature and many outliers
        # "CAPUTLG3311A2S",  # many outliers
        # "CES1021100001",  # no outliers
        # "CES4244110001",  # no outliers
        "IPN31152S",  # no outliers
        # "CIVPART",
        # "USALOLITONOSTSAM",  # many outliers
        # "MSACSR",  # many outliers
        # "UNRATE", # 4 outliers
        # "SAHMREALTIME", # many outliers
        # "CORESTICKM159SFRBATL",  # many outliers
        # "PAYEMS",  # no outliers
        # "TOTALSA",  # no outliers
        "RETAILIRSA",  # 6 outliers
        # "HOUST",  # no outliers
        # "CE16OV",  # no outliers
        "ISRATIO",  # 1 outlier
        # "DSPIC96",  # no outliers
        # "MABMM301USM189S",  # many outliers
        # "ALTSALES",  # 1 outlier
        # "DAUPSA",  # no outliers
        "RETAILIMSA",  # no outliers
        # "RSAFS",  # no outliers
        "CLF16OV",  # no outliers
        # "AISRSA",  # many outliers
        # "TOTALSL",
        # "MYAGM2USM052S",
        # "LNS14000006",  # 1 outlier
        # "CUSR0000SETA02", # 2 outliers
        # "SAHMCURRENT",  # many outliers
        "PERMIT",
        "MANEMP",
        # "USALORSGPNOSTSAM",  # many outliers
        # "STICKCPIM157SFRBATL",  # many outliers
        # "CPILFESL",
        # "BUSINV",
        "RRSFS",  # many outliers
        # "CPIUFDSL", # NO outliers
        # "BUSLOANS",  # many outliers
        # "JTSLDL",  # accuracy drops extremely with this feature and 6 outliers
        # "REVOLSL", # no outliers
        # "VMTD11", # no outliers
        # "A229RX0",  ######## makes recall down worse but improve the accuracy and no outliers
        # "CUSR0000SAH1", # no outliers
        "HSN1F",  # improve the accuracy without making recall down worse and no outliers
        # "LNS11300060",  # many outliers
        # "RSCCAS",  # accuracy drops a lot with this feature and no outliers
        # "PCETRIM12M159SFRBDAL", # many outliers
        # "LFWA64TTUSM647S",  # no outliers
        "LNS11300002",  # many outliers
        "HTRUCKSSAAR",  # no outliers
        # "EMRATIO",  # no outliers
        # "TSIFRGHTC",  # 8 outliers
        # "MRTSSM44X72USS",  # no outliers
        "CAUR",  # 4 outliers
        "CMRMTSPL",  # no outliers
        # "USAUCSFRCONDOSMSAMID",  # many outliers
        # "FLUR",  # 3 outliers
        # "UNDCONTSA",  # accuracy drops a lot with this feature and many outliers
        # "CUSR0000SEHA",  # no outliers
        # "LMJVTTUVUSM647S",  # many outliers
        # "RAILFRTCARLOADSD11",  # accuracy drops extremely without this feature and no outliers
        # "USGOVT",  # no outliers
    ],
    "COMMODITY": [
        # "Aluminum",  # accuracy drops a lot with this feature and 2 outliers
        # "Baltic Dry",  # accuracy drops a lot with this feature and many outliers
        "Brent",   # many outliers
        # "Coal",  # accuracy drops extremely with this feature and 7 outliers
        "Cocoa",  # accuracy drops extremely without this feature and no outliers
        # "Coffee",  # accuracy drops a lot with this feature and many outliers
        "Copper",  # no outliers
        # "Corn",  # accuracy drops extremely with this feature and many outliers
        # "Cotton",  # no outliers
        # "Crude Oil",  # 1 outlier
        "Ethanol",  # 1 outlier
        "Feeder Cattle",  # 3 outliers
        "Gasoline",  # no outliers
        # "Gold",  # accuracy drops a lot with this feature and no outliers
        "Heating Oil",  # 1 outlier
        "Iron Ore",  # no outliers
        # "Lead",  # accuracy drops a lot with this feature and no outliers
        # "Lean Hogs",  # 5 outliers
        # "Live Cattle",  # no outliers
        # "LME Index",  # accuracy drops extremely with this feature and no outliers
        "Lumber",  # many outliers
        "Methanol",  # no outliers
        "Natural gas",  # many outliers
        # "Nickel",  # accuracy drops extremely with this feature and many outliers
        "Propane",  # no outliers
        # "S&P GSCI",  # accuracy drops extremely with this feature and 4 outliers
        # "Silver",  # 4 outliers
        # "Soybeans",  # accuracy drops a lot with this feature and no outliers
        # "Sugar",  # accuracy drops extremely with this feature and 2 outliers
        # "Tin",  # accuracy drops a lot with this feature and 3 outliers
        # "Wheat",  # accuracy drops extremely with this feature and 1 outlier
        # "Zinc",  # accuracy drops extremely with this feature and 1 outlier
    ],
    "FI": [
        "FI_T10YFFM",
        # "FI_TB3SMFFM",  # many outliers
        "FI_GS10",  # 3 outliers
        # "FI_TB3MS", # many outliers
        "FI_EA_IRLTLT01EZM156N",
        "FI_T10Y2YM",
        # "FI_GS2",
        # "FI_T10YIEM",  # many outliers
        # "FI_T5YIEM",  # accuracy drops a lot with this feature and many outliers
        # "FI_GS30",
    ],
    "EQUITY": [
        "EQ_EA_DAX30",  # accuracy drops a lot with this feature
        "EQ_EA_CAC40",
        "EQ_EA_AEX",  # accuracy drops a lot with this feature
        # "EQ_EA_IBEX",  # many outliers
        # "EQ_Dow_Jones",  # many outliers
        # "EQ_NASDAQ",  # accuracy drops a lot with this feature and many outliers
        # "EQ_SPX",   # accuracy drops a lot with this feature and many outliers
    ],
}
US_EA_BINARY: dict = {
    "MACRO": [
        "FI_EA_IR3TIB01EZM156N",  # no outliers
        # "EA_Wage_Growth_YoY",  # accuracy drops extremely with this feature and many outliers
        # "EA_Business_Confidence",  # many outliers
        ###### "EA_Composite_PMI_Final",  # many outliers
        # "EA_Consumer_Confidence",  # 6 outliers
        # "EA_Consumer_Confidence_Final",  # accuracy drops a lot  with this feature and many outliers
        # "EA_Consumer_Confidence_Price_Trends",  # accuracy drops a lot with this feature and many outliers
        # "EA_Consumer_Inflation_Expectations",  # no outliers
        # "EA_Economic_Optimism_Index",  # accuracy drops a lot with this feature and many outliers
        "EA_Global_Construction_PMI",  # no outliers
        # "EA_Industrial_Sentiment",  # accuracy drops a lot with this feature and many outliers and many outliers
        # "EA_Manufacturing_PMI_Final",  # many outliers
        ###### "EA_Mining_Production",  # many outliers
        ###### "EA_PPI_MoM",  # many outliers
        # "EA_Producer_Prices",  # accuracy drops a lot with this feature and many outliers
        # "EA_Producer_Price_Change",  # many outliers
        ###### "EA_Retail_Sales_MoM",  # many outliers
        # "EA_Services_PMI_Final",  # accuracy drops extremely with this feature and many outliers
        # "EA_Services_Sentiment",  # accuracy drops extremely with this feature and many outliers
        "EA_Unemployment_Rate",  # 2 outliers
        # "FR_Business_Climate_Indicator",  # accuracy drops extremely with this feature and 6 outliers
        # "FR_Business_Confidence",  # accuracy drops extremely with this feature and no outliers
        ######"FR_Composite_PMI_Final",  # many outliers
        # "FR_Consumer_Confidence",  # many outliers
        "FR_GDP_Growth_Rate_QoQ_Final",  # 4 outliers
        "FR_Global_Construction_PMI",  # no outliers
        ######"FR_Manufacturing_PMI_Final",  # many outliers
        ######"FR_PPI_MoM",  # many outliers
        # "FR_Retail_Sales_MoM",  # 6 outliers
        "FR_Unemployment_Rate",  # 5 outliers
        "GR_Business_Confidence",  # accuracy drops a lot with this feature and no outliers
        "GR_Consumer_Confidence",  # no outliers
        ######"GR_Manufacturing_PMI",  # many outliers
        "GR_Unemployment_Rate",  # no outliers
        "IE_Consumer_Confidence",  # no outliers
        "IE_GDP_Growth_Rate_QoQ",  # no outliers
        # "IE_GDP_Growth_Rate_QoQ_Final",  # 1 outlier (bad)
        # "IE_Retail_Sales_MoM",  # many outliers
        # "IE_Unemployment_Rate",  # no outliers
        # "IE_Wholesale_Prices_MoM",  # 8 outliers
        # "IT_Business_Confidence",  # accuracy drops extremely with this feature and many outliers
        # "IT_Consumer_Confidence",  # accuracy drops extremely with this feature and no outliers
        # "IT_GDP_Growth_Rate_QoQ_Final",  # 9 outliers
        ######"IT_Global_Construction_PMI",  # many outliers
        ######"IT_PPI_MoM",  # many outliers
        ######"IT_Retail_Sales_MoM",  # many outliers
        # "IT_Unemployment_Rate",  # no outliers
        # "ES_Consumer_Confidence",  # accuracy drops extremely with this feature and no outliers
        # "ES_GDP_Growth_Rate_QoQ_Final",  # 6 outliers
        ######"ES_Manufacturing_PMI",  # many outliers
        # "ES_Retail_Sales_MoM",  # many outliers
        ######"ES_Services_PMI",  # many outliers
        ######"ES_Unemployment_Change",  # makes recall down better and many outliers and many outliers
        # "ES_Unemployment_Rate",  # no outliers
        ######"DE_Composite_PMI_Final",  # many outliers (bad)
        # "DE_GDP_Growth_Rate_QoQ_Final", # 8 outliers
        "DE_Global_Construction_PMI",  # 1 outlier
        # "DE_Import_Prices_MoM",  # many outliers
        # "DE_Manufacturing_PMI_Final",  # accuracy drops extremely with this feature and many outliers
        # "DE_PPI_MoM",  # many outliers
        # "DE_Retail_Sales_MoM",  # many outliers
        # "DE_Services_PMI_Final",  # many outliers
        # "DE_Unemployment_Rate",  # 7 outliers
        ######"DE_Wholesale_Prices_MoM",  # many outliers
        # "BE_Business_Confidence",  # 8 outliers
        # "BE_Consumer_Confidence",  # 9 outliers
        # "BE_GDP_Growth_Rate_QoQ_Final",  # 8 outliers
        # "BE_Retail_Sales_MoM",  # many outliers
        # "AT_Business_Confidence",  # many outliers
        # "AT_Consumer_Confidence",  # many outliers
        # "AT_GDP_Growth_Rate_QoQ_Final",  # 3 outliers
        # "AT_PPI_MoM",  # many outliers
        # "AT_Retail_Sales_MoM",  # many outliers
        ######"AT_Wholesale_Prices_MoM",  # many outliers
        "PT_Business_Confidence",  # 2 outliers
        # "PT_Consumer_Confidence",  # many outliers
        # "PT_GDP_Growth_Rate_QoQ_Final",  # many outliers
        ######"PT_PPI_MoM",  # many outliers
        # "PT_Retail_Sales_MoM",  # many outliers
        "PT_Unemployment_Rate",  # no outliers
        ######"NL_Business_Confidence",  # many outliers
        # "NL_Consumer_Confidence",  # 2 outliers
        # "NL_GDP_Growth_Rate_QoQ_Final",  # accuracy drops a lot with this feature and 4 outliers
        "NL_Unemployment_Rate",  # no outliers
        # "AHETPI",  # no outliers
        # "BOPGEXP",  # no outliers
        # "BOPGIMP",  # accuracy drops extremely with this feature and no outliers
        # "BOPGSTB",  # 1 outlier
        # "BOPGTB",  # accuracy drops extremely with this feature and no outliers
        # "BOPSEXP",  # no outliers
        # "BOPSIMP",  # no outliers
        # "BOPSTB",  # no outliers
        # "BOPTEXP",  # no outliers
        # "BOPTIMP",  # accuracy drops extremely with this feature and no outliers
        ######"CEFDFSA066MSFRBPHI",  # many outliers
        "CES0800000001",  # accuracy drops a lot without this feature and no outliers
        # "CES4244800001",  # many outliers
        "CDSP",  # no outliers
        # "CPIAUCSL",  # no outliers
        "CSUSHPISA",  # no outliers
        ######"CURRDD",  # recall down is better with this feature and many outliers
        ######"CURRSL",  # accuracy drops extremely without this feature and many outliers
        "CWSR0000SA0",  # accuracy drops a lot without this feature no outliers
        # "DEMDEPSL",  # accuracy drops a lot with this feature and many outliers
        # "DGDSRC1",  # no outliers
        # "DMANEMP",  # no outliers
        "DSPI",  # accuracy drops extremely without this feature and no outliers
        ######"FEDFUNDS",  # many outliers
        "FODSP",  # 1 outlier
        "FRBKCLMCILA",  # no outliers
        # "FRBKCLMCIM",  # accuracy drops a lot with this feature and many outliers
        ######"GACDFSA066MSFRBPHI",  # many outliers
        # "GACDISA066MSFRBNY",  # 4 outliers
        # "HPIPONM226S",  # no outliers
        # "INDPRO",  # no outliers
        "IPCONGD",  # no outliers
        "IPMANSICS",  # accuracy drops a lot without this feature and no outliers
        # "IPMINE",  # accuracy drops a lot with this feature and many outliers
        "IPUTIL",  # no outliers
        ######"IR3TIB01USM156N",  # many outliers
        # "ITMTAEM133S",  # accuracy drops extremely with this feature and 6 outliers
        # "ITXFISM133S",  # accuracy drops a lot with this feature and no outliers
        # "ITXTCIM133S",  # accuracy drops extremely with this feature and no outliers
        # "JTS1000JOL",  # 6 outliers
        # "JTS2300JOL",  # accuracy drops extremely with this feature and 6 outliers
        # "JTS3000JOL",  # many outliers
        # "JTS4000JOL",  # 5 outliers
        # "JTS4400JOL",  # 2 outliers
        # "JTS540099JOL", # accuracy drops extremely with this feature and 3 outliers
        # "JTS6000JOL",  # accuracy drops extremely with this feature and 3 outliers
        # "JTS7000JOL",  # accuracy drops extremely with this feature and 6 outliers
        # "JTS7200JOL",  # accuracy drops extremely with this feature and 7 outliers
        # "JTSJOL",  # 7 outliers
        # "JTSOSL",  # no outliers
        ######"M1REAL",  # many outliers (bad)
        ######"M1SL",  # many outliers (bad)
        ######"M2REAL",  # accuracy drops a lot without this feature and many outliers
        ######"M2SL",  # many outliers
        "MANEMP",  # accuracy drops a lot without this feature and no outliers
        ######"MCUMFN",  # many outliers
        # "MEIM683SFRBCHI",  # many outliers
        # "MNFCTRIMSA",  # many outliers
        "MNFCTRIRSA",  # 3 outliers
        # "MNFCTRMPCIMSA",  # makes recall down worse and 10 outliers
        # "MNFCTRMPCSMSA",  # many outliers
        # "MNFCTRSMSA",  # accuracy drops a lot with this feature and no outliers
        "MPCT00XXS",  # 5 outliers
        # "MPCT03XXS",  # 2 outliers
        # "MPCT04XXS",  # 2 outliers
        # "MPCT12XXS",  # many outliers
        # "MPCTNRXXS",  # 5 outliers
        # "MPCTXXXXS",  # 4 outliers
        "MPCV00XXS",  # 5 outliers
        # "NPPTTL",  # no outliers
        # "PCE",  # no outliers
        # "PCEC96",  # accuracy drops extremely with this feature and no outliers
        "PCEDG",  # accuracy drops a lot without this feature and no outliers
        # "PCEDGC96",  # accuracy drops extremely with this feature and 2 outliers
        "PCEND",  # no outliers
        # "PCENDC96",  # accuracy drops extremely with this feature and 7 outliers
        "PCEPI",  # no outliers
        "PCEPILFE",  # no outliers
        "PCES",  # no outliers
        # "PCESC96",  # accuracy drops extremely with this feature and no outliers
        ######"PCTR",  # 6 outliers
        # "PDI",  # accuracy drops extremely with this feature and no outliers
        # "PI",  # no outliers
        # "PMSAVE",  # many outliers (bad)
        "PPCDFSA066MSFRBPHI",  # 3 outliers
        # "PPCDISA066MSFRBNY",  # no outliers
        # "PRRESCONS",  # accuracy drops a lot with this feature and no outliers
        ######"PSAVERT",  # 7 outliers
        "RMFSL",
        "RPI",  # no outliers
        # "S4248SM144SCEN",  # no outliers
        # "SPCS20RSA",  # no outliers
        "STDSL",  # no outliers
        ######"TCU",  # 6 outliers
        # "TEMPHELPS",  # no outliers
        # "TLCOMCONS",  # accuracy drops a lot with this feature and no outliers
        # "TLHLTHCONS",  # accuracy drops a lot with this feature and many outliers
        # "TLHWYCONS",  # accuracy drops a lot with this feature and no outliers
        # "TLNRESCONS",  # no outliers
        # "TLRESCONS",  # accuracy drops extremely with this feature and no outliers
        # "TTLCONS",  # accuracy drops a lot with this feature and no outliers
        "USCONS",  # no outliers
        "USFIRE",  # no outliers
        "USGOOD",  # no outliers
        "USPBS",  # accuracy drops a lot without this feature and no outliers
        # "USPHCI",  # no outliers
        "USPRIV",  # no outliers
        # "USSLIND",  # many outliers (bad)
        "USTPU",  # no outliers
        "USTRADE",  # accuracy drops a lot without this feature and no outliers
        ######"W825RC1",  # accuracy drops a lot without this feature and many outliers (bad)
        "W875RX1",  # no outliers
        "WHLSLRIRSA",  # 2 outliers
        "A939RX0Q048SBEA",  # no outliers
        "B009RC1Q027SBEA",  # accuracy drops a lot without this feature and no outliers
        # "COFC",  # no outliers
        "CP",  # no outliers
        "CPATAX",  # accuracy drops a lot without this feature and no outliers
        "FPI",  # accuracy drops extremely without this feature and no outliers
        "GDI",  # no outliers
        "GDPC1",  # no outliers
        "GGSAVE",  # many outliers (bad)
        # "GPDI",  # no outliers
        "GPDIC1",  # no outliers
        "GPSAVE",  # 6 outliers
        # "GSAVE",  # no outliers
        # "IEABC",  # no outliers
        # "IEAMGS",  # no outliers
        # "IEAXGS",  # accuracy drops extremely with this feature and no outliers
        # "MDSP",   # no outliers
        # "MMMFTAQ027S",  # no outliers
        "NETFI",  # accuracy drops a lot without this feature and no outliers
        # "OUTMS",  # no outliers
        "PNFI",  # accuracy drops a lot without this feature and no outliers
        "PRFI",   # no outliers
        "RSAHORUSQ156S",  # no outliers
        # "TDSP",  # no outliers
        # "ULCMFG",  # many outliers
        # "W207RC1Q156SBEA", # 2 outliers
        ######"W986RC1Q027SBEA",  # accuracy drops a lot without this feature and many outliers (bad)
        # "IPG211111CS",  # accuracy drops a lot with this feature and no outliers
        # "IPG3361T3S",  # accuracy drops a lot with this feature and no outliers
        # "IPMAN",  # accuracy drops a lot with this feature and no outliers
        # "IPN3311A2RS",  # accuracy drops a lot with this feature and many outliers
        # "CAPUTLG3311A2S",  # many outliers
        # "CES1021100001",  # no outliers
        # "CES4244110001",  # no outliers
        "IPN31152S",  # no outliers
        # "CIVPART",
        # "USALOLITONOSTSAM",  # many outliers
        # "MSACSR",  # many outliers
        # "UNRATE", # 4 outliers
        # "SAHMREALTIME", # many outliers
        # "CORESTICKM159SFRBATL",  # many outliers
        # "PAYEMS",  # no outliers
        # "TOTALSA",  # no outliers
        ######"RETAILIRSA",  # 6 outliers
        # "HOUST",  # no outliers
        # "CE16OV",  # no outliers
        "ISRATIO",  # 1 outlier
        # "DSPIC96",  # no outliers
        # "MABMM301USM189S",  # many outliers
        # "ALTSALES",  # 1 outlier
        # "DAUPSA",  # no outliers
        "RETAILIMSA",  # no outliers
        # "RSAFS",  # no outliers
        "CLF16OV",  # no outliers
        # "AISRSA",  # many outliers
        # "TOTALSL",
        # "MYAGM2USM052S",
        # "LNS14000006",  # 1 outlier
        # "CUSR0000SETA02", # 2 outliers
        # "SAHMCURRENT",  # many outliers
        "PERMIT",  # no outliers
        "MANEMP",  # no outliers
        # "USALORSGPNOSTSAM",  # many outliers
        # "STICKCPIM157SFRBATL",  # many outliers
        # "CPILFESL",  # no outliers
        # "BUSINV",  # no outliers
        ######"RRSFS",  # many outliers
        # "CPIUFDSL", # no outliers
        # "BUSLOANS",  # many outliers
        # "JTSLDL",  # accuracy drops extremely with this feature and 6 outliers
        # "REVOLSL", # no outliers
        # "VMTD11", # no outliers
        # "A229RX0",  ######## makes recall down worse but improve the accuracy and no outliers
        # "CUSR0000SAH1", # no outliers
        "HSN1F",  # improve the accuracy without making recall down worse and no outliers
        # "LNS11300060",  # many outliers
        # "RSCCAS",  # accuracy drops a lot with this feature and no outliers
        # "PCETRIM12M159SFRBDAL", # many outliers
        # "LFWA64TTUSM647S",  # no outliers
        ######"LNS11300002",  # many outliers
        "HTRUCKSSAAR",  # no outliers
        # "EMRATIO",  # no outliers
        # "TSIFRGHTC",  # 8 outliers
        # "MRTSSM44X72USS",  # no outliers
        "CAUR",  # 4 outliers
        "CMRMTSPL",  # no outliers
        # "USAUCSFRCONDOSMSAMID",  # many outliers
        # "FLUR",  # 3 outliers
        # "UNDCONTSA",  # accuracy drops a lot with this feature and many outliers
        # "CUSR0000SEHA",  # no outliers
        # "LMJVTTUVUSM647S",  # many outliers
        # "RAILFRTCARLOADSD11",  # accuracy drops extremely without this feature and no outliers
        # "USGOVT",  # no outliers
    ],
    "COMMODITY": [
        # "Aluminum",  # accuracy drops a lot with this feature and 2 outliers
        # "Baltic Dry",  # accuracy drops a lot with this feature and many outliers
        ######"Brent",   # many outliers
        # "Coal",  # accuracy drops extremely with this feature and 7 outliers
        "Cocoa",  # accuracy drops extremely without this feature and no outliers
        # "Coffee",  # accuracy drops a lot with this feature and many outliers
        "Copper",  # no outliers
        # "Corn",  # accuracy drops extremely with this feature and many outliers
        # "Cotton",  # no outliers
        # "Crude Oil",  # 1 outlier
        "Ethanol",  # 1 outlier
        "Feeder Cattle",  # 3 outliers
        "Gasoline",  # no outliers
        # "Gold",  # accuracy drops a lot with this feature and no outliers
        "Heating Oil",  # 1 outlier
        "Iron Ore",  # no outliers
        # "Lead",  # accuracy drops a lot with this feature and no outliers
        # "Lean Hogs",  # 5 outliers
        # "Live Cattle",  # no outliers
        # "LME Index",  # accuracy drops extremely with this feature and no outliers
        ######"Lumber",  # many outliers
        "Methanol",  # no outliers
        ######"Natural gas",  # many outliers
        # "Nickel",  # accuracy drops extremely with this feature and many outliers
        "Propane",  # no outliers
        # "S&P GSCI",  # accuracy drops extremely with this feature and 4 outliers
        # "Silver",  # 4 outliers
        # "Soybeans",  # accuracy drops a lot with this feature and no outliers
        # "Sugar",  # accuracy drops extremely with this feature and 2 outliers
        # "Tin",  # accuracy drops a lot with this feature and 3 outliers
        # "Wheat",  # accuracy drops extremely with this feature and 1 outlier
        # "Zinc",  # accuracy drops extremely with this feature and 1 outlier
    ],
    "FI": [
        ######"FI_T10YFFM",  # many outliers
        ######"FI_TB3SMFFM",  # many outliers
        "FI_GS10",  # 3 outliers
        ######"FI_TB3MS",  # many outliers
        "FI_EA_IRLTLT01EZM156N",  # no outliers
        "FI_T10Y2YM",  # no outliers
        # "FI_GS2",  # no outliers
        # "FI_T10YIEM",  # many outliers
        # "FI_T5YIEM",  # accuracy drops a lot with this feature and many outliers
        # "FI_GS30",  # no outliers
    ],
    "EQUITY": [
        "EQ_EA_DAX30",  # accuracy drops a lot with this feature and no outliers
        "EQ_EA_CAC40",  # no outliers
        "EQ_EA_AEX",  # accuracy drops a lot with this feature and no outliers
        ######"EQ_EA_IBEX",  # many outliers
        ######"EQ_Dow_Jones",  # many outliers
        # "EQ_NASDAQ",  # accuracy drops a lot with this feature and many outliers
        # "EQ_SPX",   # accuracy drops a lot with this feature and many outliers
    ],
}

