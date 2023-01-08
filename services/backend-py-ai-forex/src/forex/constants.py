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
# MOVING AVERAGE
SMA: bool = False
SMA_WINDOW: int = 20
SMA_PLOT: bool = True
# PLOT
PLOT_VP_THREE_MONTHS: bool = False
PLOT_VP_SIX_MONTHS: bool = False
PLOT_VP_NINE_MONTHS: bool = False
PLOT_VP_TWELVE_MONTHS: bool = False

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

