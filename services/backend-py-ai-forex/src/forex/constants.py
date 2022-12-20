from pathlib import Path

BASE_PATH = Path(__file__).parent

DATA_PATH: str = f"{BASE_PATH}/data"
MACRO_DATA: str = f"{DATA_PATH}/macro_data-all_data_2.0.csv"
MACRO_DELAY_SOURCE: str = f"{DATA_PATH}/delay_macro.csv"
TRAIN_DEV_DATA: str = f"{DATA_PATH}/ML_Data_29un2022-train+dev.csv"
TRAIN_DATA: str = f"{DATA_PATH}/ML_Data_29un2022-train.csv"
DEV_DATA: str = f"{DATA_PATH}/ML_Data_29un2022-train.csv"
TEST_DATA: str = f"{DATA_PATH}/ML_Data_29un2022-test.csv"

TRAINING_WITH_TRAIN_DEV_DATA: bool = True
MODEL_TYPE = "logistic"  # logistic, SVM, random_forest, decision_tree, xgboost
ADD_MOVING_AVERAGE: bool = True
ADD_MACRO: bool = True
ADD_COMMODITY: bool = True
ADD_FI: bool = True
ADD_EQUITY: bool = True
WINDOW_PROBABILITY_ANALYSIS_LONGEST: int = 6
WINDOW_PROBABILITY_ANALYSIS_NUMBER: int = 3
PLOT_VP_SIX_MONTHS: bool = False

# BINARY CLASSIFICATION
BINARY_CLASSIFICATION: bool = False
BINARY_LABELS: list = ["down", "up"]
BINARY_THRESHOLD_FOREX: float = 0.1
BINARY_FEATURES: dict = {
    "MACRO": [
        # "AHETPI",
        "BOPGEXP",
        "BOPGIMP",
        "BOPGSTB",
        "BOPGTB",
        # "BOPSEXP",
        # "BOPSIMP",
        "BOPSTB",
        "BOPTEXP",
        # "BOPTIMP", --
        # "CAPUTLG3311A2S",
        "CEFDFSA066MSFRBPHI",
        # "CES0800000001",
        # "CES1021100001",
        "CES4244110001",
        "CES4244800001",
        "CPIAUCSL",
        # "CSUSHPISA",
        # "CURRDD", --
        "CURRSL",
        "CWSR0000SA0",
        "DEMDEPSL",
        "DGDSRC1",
        # "DMANEMP",
        "DSPI",
        # "FEDFUNDS",
        # "FRBKCLMCILA", --
        "FRBKCLMCIM",
        "GACDFSA066MSFRBPHI",
        # "GACDISA066MSFRBNY",
        "HPIPONM226S",  # accuracy drops a lot without this feature
        # "INDPRO",
        # "IPCONGD",
        "IPG211111CS",
        # "IPG3361T3S",
        # "IPMAN",
        # "IPMANSICS",
        # "IPMINE",
        "IPN31152S",
        # "IPN3311A2RS",
        # "IPUTIL",
        # "IR3TIB01USM156N",
        "ITMTAEM133S",  # accuracy drops without this feature
        "ITXFISM133S",
        # "ITXTCIM133S", --
        "JTS1000JOL",
        "JTS2300JOL",  # accuracy drops without this feature
        "JTS3000JOL",
        "JTS4000JOL",
        "JTS4400JOL",  # accuracy drops without this feature
        "JTS540099JOL",  # accuracy drops without this feature
        "JTS6000JOL",
        "JTS7000JOL",
        # "JTS7200JOL", --
        "JTSJOL",
        "JTSOSL",
        "M1REAL",
        # "M1SL",
        "M2REAL",
        "M2SL",
        "MANEMP",  # accuracy drops a lot without this feature
        "MCUMFN",
        "MEIM683SFRBCHI",
        "MNFCTRIMSA",
        # "MNFCTRIRSA", --
        "MNFCTRMPCIMSA",
        "MNFCTRMPCSMSA",  # accuracy drops without this feature
        # "MNFCTRSMSA",
        "MPCT00XXS",
        "MPCT03XXS",
        # "MPCT04XXS",
        "MPCT12XXS",
        "MPCTNRXXS",
        "MPCTXXXXS",
        "MPCV00XXS",
        # "NPPTTL", --
        # "PCE",
        "PCEC96",
        "PCEDG",
        "PCEDGC96",
        # "PCEND", --
        "PCENDC96",
        # "PCEPI",
        # "PCEPILFE",
        # "PCES",
        "PCESC96",
        "PCTR",
        "PDI",  # accuracy drops a lot without this feature
        "PI",
        # "PMSAVE",
        # "PPCDFSA066MSFRBPHI",
        "PPCDISA066MSFRBNY",  # accuracy drops a lot without this feature
        # "PRRESCONS",
        "PSAVERT",  # acuracy drops a lot without this feature
        "RMFSL",  # accuracy drops a lot without this feature
        "RPI",
        "S4248SM144SCEN",
        # "SPCS20RSA",
        "STDSL",  # accuracy drops without this feature
        # "TCU",
        "TEMPHELPS",
        "TLCOMCONS",  # accuracy drops a lot without this feature
        "TLHLTHCONS",  # accuracy drops extremely without this feature
        # "TLHWYCONS",
        # "TLNRESCONS",
        "TLRESCONS",
        "TTLCONS",  # accuracy drops a lot without this feature
        # "USCONS",
        "USFIRE",
        # "USGOOD",
        "USPBS",
        "USPHCI",
        "USPRIV",
        # "USSLIND", --
        # "USTPU", --
        # "USTRADE",
        "W825RC1",
        # "W875RX1",
        # "WHLSLRIRSA",
        # "A939RX0Q048SBEA",
        # "B009RC1Q027SBEA",
        # "CDSP", --
        "COFC",
        # "CP", --
        "CPATAX",
        # "FODSP",
        "FPI",  # accuracy drops a lot without this feature
        "GDI",
        "GDPC1",
        # "GGSAVE",
        "GPDI",
        # "GPDIC1", --
        # "GPSAVE", --
        "GSAVE",
        # "IEABC",
        "IEAMGS",
        "IEAXGS",
        # "MDSP", # --
        # "MMMFTAQ027S",
        # "NETFI",
        # "OUTMS",  # --
        # "PNFI",   # --
        # "PRFI",
        # "RSAHORUSQ156S",
        # "TDSP",
        "ULCMFG",
        # "W207RC1Q156SBEA",
        # "W986RC1Q027SBEA",
    ],
    "COMMODITY": [
        # "Aluminum",  # accuracy drops with this feature
        # "Baltic Dry",
        # "Brent",
        # "Coal",
        # "Cocoa",
        # "Coffee",  # accuracy drops a lot with this feature
        "Copper",
        # "Corn",  # accuracy drops a lot with this feature
        "Cotton",  # --
        "Crude Oil",
        # "Ethanol",
        # "Feeder Cattle",
        "Gasoline",  # --
        # "Gold",
        "Heating Oil",
        "Iron Ore",
        # "Lead",
        "Lean Hogs",
        # "Live Cattle",
        "LME Index",
        # "Lumber",
        "Methanol",
        "Natural gas",
        "Nickel",  # accuracy drops a lot without this feature
        "Propane",  # --
        # "S&P GSCI",
        "Silver",
        # "Soybeans",
        # "Sugar",
        # "Tin",
        # "Wheat",
        # "Zinc",
    ],
    "FI": [
        # "Adj Close",
        # "T10Y2YM",
        # "T10YFFM",
        # "TB3SMFFM",
        # "T10YIEM",
        # "T5YIEM",
        # "GS10",
        "GS30",  # --
        # "GS2",
        "TB3MS",
    ],
    # "MONEY": ["Adj Close"],
    "EQUITY": [
        # "Adj Close - Dow Jones",
        "Adj Close - NASDAQ",
        "Spx - Adj Close"
    ],
}

# US_EA BINARY CLASSIFICATION
US_EA_CLASSIFICATION: bool = False
US_EA_BINARY_FEATURES: dict = {
    "MACRO": [
        # "EA_Business_Confidence", #
        # "EA_Business_Confidence2", #
        # "EA_Composite_PMI_Final", #
        "EA_Consumer_Confidence",
        # "EA_Consumer_Confidence_Final", #
        "EA_Consumer_Confidence_Price_Trends",
        "EA_Consumer_Inflation_Expectations",
        "EA_Economic_Optimism_Index",
        # "EA_Global_Construction_PMI", #
        "EA_Industrial_Sentiment",
        # "EA_Industrial_Sentiment2" #
        # "EA_Manufacturing_PMI_Final", #
        "EA_Mining_Production",
        # "EA_PPI_MoM", #
        "EA_Producer_Prices",
        "EA_Producer_Price_Change",
        # "EA_Retail_Sales_MoM", #
        "EA_Services_PMI_Final",
        "EA_Services_Sentiment",
        "EA_Unemployment_Rate",
        "EA_Wage_Growth_YoY",
        # "ES_Business_Confidence", #
        "ES_Consumer_Confidence",
        # "ES_GDP_Growth_Rate_QoQ_Final",
        # "ES_Manufacturing_PMI", #
        # "ES_Retail_Sales_MoM", #
        # "ES_Services_PMI", #
        # "ES_Unemployment_Change", #
        "ES_Unemployment_Rate",
        "FI_Business_Confidence",
        "FI_Consumer_Confidence",
        "FI_GDP_Growth_Rate_QoQ_Final",
        "FI_Unemployment_Rate",
        "FR_Business_Climate_Indicator",
        # "FR_Business_Confidence", #
        # "FR_Composite_PMI_Final", #
        # "FR_Consumer_Confidence", #
        # "FR_GDP_Growth_Rate_QoQ_Final", #
        # "FR_Global_Construction_PMI", #
        # "FR_Manufacturing_PMI_Final", #
        # "FR_PPI_MoM", #
        # "FR_Retail_Sales_MoM", #
        # "FR_Services_PMI_Final", #
        "FR_Unemployment_Rate",
        "GR_Business_Confidence",
        # "GR_Consumer_Confidence", #
        # "GR_Manufacturing_PMI_Final", #
        "GR_Unemployment_Rate",
        "IE_Consumer_Confidence",
        # "IE_GDP_Growth_Rate_QoQ", #
        # "IE_GDP_Growth_Rate_QoQ_Final", #
        "IE_Retail_Sales_MoM",
        "IE_Unemployment_Rate",
        "IE_Wholesale_Prices_MoM",
        "IT_Business_Confidence",
        # "IT_Consumer_Confidence",
        # "IT_GDP_Growth_Rate_QoQ_Final", #
        # "IT_Global_Construction_PMI", #
        "IT_PPI_MoM",
        # "IT_Retail_Sales_MoM", #
        "IT_Unemployment_Rate",
        # "AHETPI",
        "BOPGEXP",
        "BOPGIMP",
        "BOPGSTB",
        "BOPGTB",
        # "BOPSEXP",
        # "BOPSIMP",
        # "BOPSTB",
        "BOPTEXP",
        # "BOPTIMP", #
        # "CAPUTLG3311A2S",
        "CEFDFSA066MSFRBPHI",
        "CES0800000001",
        # "CES1021100001",
        # "CES4244110001", #
        # "CES4244800001", #
        # "CPIAUCSL", #
        # "CSUSHPISA",
        # "CURRDD", #
        "CURRSL",
        "CWSR0000SA0",
        "DEMDEPSL",
        "DGDSRC1",
        # "DMANEMP",
        "DSPI",
        # "FEDFUNDS", #
        # "FRBKCLMCILA", #
        "FRBKCLMCIM",
        "GACDFSA066MSFRBPHI",
        # "GACDISA066MSFRBNY",
        "HPIPONM226S",  # accuracy drops a lot without this feature
        # "INDPRO",
        # "IPCONGD",
        "IPG211111CS",
        # "IPG3361T3S",
        # "IPMAN", #
        # "IPMANSICS",
        # "IPMINE",
        "IPN31152S",
        # "IPN3311A2RS",
        # "IPUTIL",
        # "IR3TIB01USM156N", #
        "ITMTAEM133S",  # accuracy drops without this feature
        "ITXFISM133S",
        # "ITXTCIM133S", #
        "JTS1000JOL",
        "JTS2300JOL",  # accuracy drops without this feature
        "JTS3000JOL",
        "JTS4000JOL",
        "JTS4400JOL",  # accuracy drops without this feature
        "JTS540099JOL",  # accuracy drops without this feature
        "JTS6000JOL",
        "JTS7000JOL",
        # "JTS7200JOL",
        # "JTSJOL",
        "JTSOSL",
        "M1REAL",
        # "M1SL",
        "M2REAL",
        "M2SL",
        "MANEMP",
        "MCUMFN",
        "MEIM683SFRBCHI",
        "MNFCTRIMSA",
        # "MNFCTRIRSA", #
        # "MNFCTRMPCIMSA", #
        # "MNFCTRMPCSMSA", #
        # "MNFCTRSMSA",
        "MPCT00XXS",
        "MPCT03XXS",
        # "MPCT04XXS",
        "MPCT12XXS",
        # "MPCTNRXXS", #
        "MPCTXXXXS",
        # "MPCV00XXS", #
        # "NPPTTL",
        # "PCE",
        # "PCEC96", #
        "PCEDG",
        "PCEDGC96",
        # "PCEND",
        "PCENDC96",
        # "PCEPI", #
        # "PCEPILFE", #
        # "PCES",
        "PCESC96",
        "PCTR",
        # "PDI", #
        "PI",
        # "PMSAVE",
        # "PPCDFSA066MSFRBPHI", #
        "PPCDISA066MSFRBNY",
        # "PRRESCONS", #
        "PSAVERT",
        "RMFSL",
        "RPI",
        "S4248SM144SCEN",
        # "SPCS20RSA", #
        "STDSL",
        # "TCU",
        "TEMPHELPS",
        "TLCOMCONS",
        "TLHLTHCONS",
        # "TLHWYCONS", #
        # "TLNRESCONS",
        # "TLRESCONS", #
        "TTLCONS",  # accuracy drops a lot without this feature
        # "USCONS",
        "USFIRE",
        # "USGOOD",
        "USPBS",
        "USPHCI",
        "USPRIV",
        # "USSLIND", #
        # "USTPU",
        # "USTRADE", #
        "W825RC1",
        # "W875RX1", #
        # "WHLSLRIRSA", #
        # "A939RX0Q048SBEA",
        # "B009RC1Q027SBEA",
        # "CDSP", #
        "COFC",
        # "CP", #
        "CPATAX",
        # "FODSP",
        "FPI",  # accuracy drops a lot without this feature
        "GDI",
        "GDPC1",
        # "GGSAVE", #
        "GPDI",
        # "GPDIC1", #
        # "GPSAVE", #
        # "GSAVE", #
        # "IEABC", #
        # "IEAMGS", #
        # "IEAXGS",
        # "MDSP", #
        # "MMMFTAQ027S",
        # "NETFI",
        # "OUTMS", #
        # "PNFI",  #
        # "PRFI",
        # "RSAHORUSQ156S",
        # "TDSP",
        "ULCMFG",
        "W207RC1Q156SBEA",  ##
        # "W986RC1Q027SBEA",
        # "AT_Business_Confidence",
        # "AT_Consumer_Confidence",
        # "AT_GDP_Growth_Rate_QoQ_Final",  #
        # "AT_PPI_MoM",  #
        # "AT_Retail_Sales_MoM",
        # "AT_Wholesale_Prices_MoM",  #
        "BE_Business_Confidence",  ##
        # "BE_Consumer_Confidence",
        # "BE_GDP_Growth_Rate_QoQ_Final",
        # "BE_Retail_Sales_MoM",
        # "DE_Composite_PMI_Final",  #
        # "DE_GDP_Growth_Rate_QoQ_Final",  #
        # "DE_Global_Construction_PMI",  #
        # "DE_Import_Prices_MoM",
        # "DE_Manufacturing_PMI_Final",
        "DE_PPI_MoM",  #
        "DE_Retail_Sales_MoM",  #
        "DE_Services_PMI_Final",  #
        "DE_Unemployment_Rate",  #
        # "DE_Wholesale_Prices_MoM",  #
        # "NL_Business_Confidence",  #
        # "NL_Consumer_Confidence",  #
        # "NL_GDP_Growth_Rate_QoQ_Final",  #
        # "NL_Unemployment_Rate",  #
        # "PT_Business_Confidence",  #
        # "PT_Consumer_Confidence",
        # "PT_GDP_Growth_Rate_QoQ_Final",
        # "PT_PPI_MoM",  #
        # "PT_Retail_Sales_MoM",  #
        # "PT_Unemployment_Rate"
    ],
    "COMMODITY": [
        # "Aluminum",  # accuracy drops with this feature
        "Baltic Dry",
        # "Brent",
        "Coal",
        "Cocoa",
        # "Coffee",  # accuracy drops a lot with this feature
        # "Copper",
        # "Corn",  # accuracy drops a lot with this feature
        # "Cotton", #
        # "Crude Oil",
        # "Ethanol",
        "Feeder Cattle",
        # "Gasoline", #
        "Gold",
        # "Heating Oil", #
        "Iron Ore",
        "Lead",
        "Lean Hogs",
        "Live Cattle",
        "LME Index",
        "Lumber",
        "Methanol",
        "Natural gas",
        "Nickel",  # accuracy drops a lot without this feature
        # "Propane", #
        "S&P GSCI",
        "Silver",
        "Soybeans",
        "Sugar",
        "Tin",
        "Wheat",
        "Zinc",
    ],
    "FI": [
        # "FI_T10Y2YM",
        # "FI_T10YFFM",
        # "FI_TB3SMFFM",
        # "FI_T10YIEM",
        # "FI_T5YIEM",
        # "FI_GS10",
        # "FI_GS30",
        # "FI_GS2",
        "FI_TB3MS",
        "FI_EA_IRLTLT01EZM156N",
        "FI_EA_IR3TIB01EZM156N",
        "FI_EA_10Y-Minus-3M"
    ],
    "EQUITY": [
        # "EQ_Dow_Jones",
        # "EQ_NASDAQ",
        # "EQ_SPX",
        "EQ_EA_AEX",
        "EQ_EA_CAC40",
        "EQ_EA_DAX30",
        "EQ_EA_IBEX"
    ],
}

TEST: bool = True
US_EA_BINARY_FEATURES2: dict = {
    "MACRO": [
        "EA_Mining_Production",
        # "AHETPI",
        "CEFDFSA066MSFRBPHI",
        "CES0800000001",  # accuracy drops a lot without this feature
        # "CPIAUCSL",
        "CURRDD",
        "CURRSL",  # accuracy drops extremely without this feature
        "CWSR0000SA0",  # accuracy drops a lot without this feature
        # "DEMDEPSL",  # accuracy drops a lot with this feature
        # "DGDSRC1",
        # "DMANEMP",
        "DSPI",  # accuracy drops extremely without this feature
        "FEDFUNDS",
        "GACDFSA066MSFRBPHI",
        # "INDPRO",
        "IPCONGD",
        "IPMANSICS",  # accuracy drops a lot without this feature
        # "IPMINE",  # accuracy drops a lot with this feature
        "IPUTIL",
        "IR3TIB01USM156N",
        "M1REAL",
        "M1SL",
        "M2REAL",  # accuracy drops a lot without this feature
        "M2SL",
        "MANEMP",  # accuracy drops a lot without this feature
        # "PCE",
        "PCEDG",  # accuracy drops a lot without this feature
        "PCEND",
        "PCEPI",
        "PCEPILFE",
        "PCES",
        "PCTR",
        # "PDI",  # accuracy drops extremely with this feature
        # "PI",
        # "PMSAVE",
        "PPCDFSA066MSFRBPHI",
        "PSAVERT",
        "RPI",
        "STDSL",
        "TCU",
        "USCONS",
        "USFIRE",
        "USGOOD",
        "USPBS",  # accuracy drops a lot without this feature
        "USPRIV",
        "USTPU",
        "USTRADE",  # accuracy drops a lot without this feature
        "W825RC1",  # accuracy drops a lot without this feature
        "W875RX1",
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
        "NETFI",  # accuracy drops a lot without this feature
        "PNFI",  # accuracy drops a lot without this feature
        "PRFI",
        "W986RC1Q027SBEA"  # accuracy drops a lot without this feature
        # "IPG211111CS",  # accuracy drops a lot with this feature
        # "IPG3361T3S",  # accuracy drops a lot with this feature
        # "IPMAN",  # accuracy drops a lot with this feature
        # "IPN3311A2RS",  # accuracy drops a lot with this feature
        # "CAPUTLG3311A2S",
        # "CES1021100001",
        # "CES4244110001",
        # "IPN31152S",
        # "NL_Business_Confidence",  # accuracy drops a lot with this feature
        # "NL_Consumer_Confidence",  # accuracy drops a lot with this feature
        # "NL_GDP_Growth_Rate_QoQ_Final", # accuracy drops a lot with this feature
        # "NL_Unemployment_Rate",  # accuracy drops a lot with this feature
        # "PT_Business_Confidence",
        # "PT_Consumer_Confidence",
        # "PT_GDP_Growth_Rate_QoQ_Final",
        # "PT_PPI_MoM",
        # "PT_Retail_Sales_MoM",
        # "PT_Unemployment_Rate",
        # "AT_Business_Confidence",
        # "AT_Consumer_Confidence",
        # "AT_GDP_Growth_Rate_QoQ_Final",
        # "AT_PPI_MoM",
        # "AT_Retail_Sales_MoM",
        # "AT_Wholesale_Prices_MoM",
        # "BE_Business_Confidence",
        # "BE_Consumer_Confidence",
        # "BE_GDP_Growth_Rate_QoQ_Final",
        # "BE_Retail_Sales_MoM",
        # "DE_Composite_PMI_Final",
        # "DE_GDP_Growth_Rate_QoQ_Final",
        # "DE_Global_Construction_PMI",
        # "DE_Import_Prices_MoM",
        # "DE_Manufacturing_PMI_Final",
        # "DE_PPI_MoM",
        # "DE_Retail_Sales_MoM",
        # "DE_Services_PMI_Final",
        # "DE_Unemployment_Rate",
        # "DE_Wholesale_Prices_MoM",
        # "ES_Consumer_Confidence",
        # "ES_GDP_Growth_Rate_QoQ_Final",
        # "ES_Manufacturing_PMI",
        # "ES_Retail_Sales_MoM",
        # "ES_Services_PMI",
        # "ES_Unemployment_Change",
        # "ES_Unemployment_Rate",
        # "FI_Business_Confidence",
        # "FI_Consumer_Confidence",
        # "FI_GDP_Growth_Rate_QoQ_Final",
        # "FI_Unemployment_Rate",
        # "FR_Business_Climate_Indicator",
        # "FR_Business_Confidence",
        # "FR_Composite_PMI_Final",
        # "FR_Consumer_Confidence",
        # "FR_GDP_Growth_Rate_QoQ_Final",
        # "FR_Global_Construction_PMI",
        # "FR_Manufacturing_PMI_Final",
        # "FR_PPI_MoM",
        # "FR_Retail_Sales_MoM",
        # "FR_Services_PMI",
        # "FR_Unemployment_Rate",
        # "GR_Business_Confidence",
        # "GR_Consumer_Confidence",
        # "GR_Manufacturing_PMI",
        # "GR_Unemployment_Rate",
        # "IE_Consumer_Confidence",
        # "IE_GDP_Growth_Rate_QoQ",
        # "IE_GDP_Growth_Rate_QoQ_Final",
        # "IE_Retail_Sales_MoM",
        # "IE_Unemployment_Rate",
        # "IE_Wholesale_Prices_MoM",
        # "IT_Business_Confidence",
        # "IT_Consumer_Confidence",
        # "IT_GDP_Growth_Rate_QoQ_Final",
        # "IT_Global_Construction_PMI",
        # "IT_PPI_MoM",
        # "IT_Retail_Sales_MoM",
        # "IT_Unemployment_Rate",
        # "EA_Business_Confidence", ##############
        # "EA_Business_Confidence2", #
        # "EA_Composite_PMI_Final", #
        # "EA_Consumer_Confidence",  ###
        # "EA_Consumer_Confidence_Final", #
        # "EA_Consumer_Confidence_Price_Trends",
        # "EA_Consumer_Inflation_Expectations",
        # "EA_Economic_Optimism_Index",
        # "EA_Global_Construction_PMI", #
        # "EA_Industrial_Sentiment",
        # "EA_Industrial_Sentiment2" #
        # "EA_Manufacturing_PMI_Final", #
        # "EA_Mining_Production",
        # "EA_PPI_MoM", #
        # "EA_Producer_Prices",
        # "EA_Producer_Price_Change",
        # "EA_Retail_Sales_MoM", #
        # "EA_Services_PMI_Final",
        # "EA_Services_Sentiment",
        # "EA_Unemployment_Rate",
        # "EA_Wage_Growth_YoY",
    ],
    "COMMODITY": [
        # "Aluminum",  # accuracy drops a lot with this feature
        # "Baltic Dry",  # accuracy drops a lot with this feature
        "Brent",
        # "Coal", # accuracy drops extremely with this feature
        "Cocoa",
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
        "FI_T10YFFM",  #
        "FI_TB3SMFFM",
        "FI_GS10",  #
        "FI_TB3MS",  #
        "FI_EA_IRLTLT01EZM156N",
        "FI_T10Y2YM",
        # "FI_GS2",
        # "FI_T10YIEM",
        # "FI_T5YIEM",   # accuracy drops a lot with this feature
        # "FI_GS30",
        "FI_EA_IR3TIB01EZM156N",
        # "FI_EA_10Y-Minus-3M",
    ],
    "EQUITY": [
        "EQ_EA_DAX30",
        "EQ_EA_CAC40",
        "EQ_EA_AEX",
        "EQ_EA_IBEX",
        # "EQ_Dow_Jones",
        # "EQ_NASDAQ",  # accuracy drops a lot with this feature
        # "EQ_SPX",   # accuracy drops a lot with this feature
    ],
}

