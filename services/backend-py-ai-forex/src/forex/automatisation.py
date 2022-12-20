import unittest

from parameterized import parameterized


def assign_boolean_property(original: list, new_key):
    aggregated = []
    if len(original) == 0:
        original = [{new_key: None}]
    for obj in original:
        for b in [0, 1]:
            obj[new_key] = True if b == 1 else False
            aggregated.append(obj.copy())
    return aggregated


def compute_truth_table(initial: list) -> list:
    truth_table = []
    for e in range(len(initial)):
        truth_table = assign_boolean_property(truth_table.copy(), initial[e])
    return truth_table


class TruthTableTest(unittest.TestCase):
    @parameterized.expand(
        [
            [{"one": ["hello"]}, [
                {"hello": False},
                {"hello": True}
            ]],
            [{"one": ["hello"], "two": ["bye"]}, [
                {"hello": False, "bye": False},
                {"hello": False, "bye": True},
                {"hello": True, "bye": False},
                {"hello": True, "bye": True},
            ]],
            [{"one": ["hello"], "two": ["wazza"], "three": ["bye"]}, [
                {"hello": False, "wazza": False, "bye": False},
                {"hello": False, "wazza": False, "bye": True},
                {"hello": False, "wazza": True, "bye": False},
                {"hello": False, "wazza": True, "bye": True},
                {"hello": True, "wazza": False, "bye": False},
                {"hello": True, "wazza": False, "bye": True},
                {"hello": True, "wazza": True, "bye": False},
                {"hello": True, "wazza": True, "bye": True},
            ]],

            # keep_macro = ["HPIPONM226S", "ITMTAEM133S", "JTS2300JOL", "JTS4400JOL", "JTS540099JOL", "MANEMP", "MNFCTRMPCSMSA",
            #               "PDI", "PPCDISA066MSFRBNY", "PSAVERT,  "RMFSL", "STDSL", "TLCOMCONS", "TLHLTHCONS", "TTLCONS", "FPI", "PPIYOYFYOYFYF"]
            # lose_commodity = ["Aluminum", "Coffee", "Corn"]
            # keep_commodity = ["Nickel"]

            [{
                "MACRO": [
                    "AHETPI",
                    "BOPGEXP",
                    "BOPGIMP",
                    "BOPGSTB",
                    "BOPGTB",
                    "BOPSEXP",
                    "BOPSIMP",
                    "BOPSTB",
                    "BOPTEXP",
                    "BOPTIMP",
                    "CAPUTLG3311A2S",
                    "CEFDFSA066MSFRBPHI",
                    "CES0800000001",
                    "CES1021100001",
                    "CES4244110001",
                    "CES4244800001",
                    "CPIAUCSL",
                    "CSUSHPISA",
                    "CURRDD",
                    "CURRSL",
                    "CWSR0000SA0",
                    "DEMDEPSL",
                    "DGDSRC1",
                    "DMANEMP",
                    "DSPI",
                    "FEDFUNDS",
                    "FRBKCLMCILA",
                    "FRBKCLMCIM",
                    "GACDFSA066MSFRBPHI",
                    "GACDISA066MSFRBNY",
                    "INDPRO",
                    "IPCONGD",
                    "IPG211111CS",
                    "IPG3361T3S",
                    "IPMAN",
                    "IPMANSICS",
                    "IPMINE",
                    "IPN31152S",
                    "IPN3311A2RS",
                    "IPUTIL",
                    "IR3TIB01USM156N",
                    "ITXFISM133S",
                    "ITXTCIM133S",
                    "JTS1000JOL",
                    "JTS3000JOL",
                    "JTS4000JOL",
                    "JTS6000JOL",
                    "JTS7000JOL",
                    "JTS7200JOL",
                    "JTSJOL",
                    "JTSOSL",
                    "M1REAL",
                    "M1SL",
                    "M2REAL",
                    "M2SL",
                    "MCUMFN",
                    "MEIM683SFRBCHI",
                    "MNFCTRIMSA",
                    "MNFCTRIRSA",
                    "MNFCTRMPCIMSA",
                    "MNFCTRSMSA",
                    "MPCT00XXS",
                    "MPCT03XXS",
                    "MPCT04XXS",
                    "MPCT12XXS",
                    "MPCTNRXXS",
                    "MPCTXXXXS",
                    "MPCV00XXS",
                    "NPPTTL",
                    "PCE",
                    "PCEC96",
                    "PCEDG",
                    "PCEDGC96",
                    "PCEND",
                    "PCENDC96",
                    "PCEPI",
                    "PCEPILFE",
                    "PCES",
                    "PCESC96",
                    "PCTR",
                    "PI",
                    "PMSAVE",
                    "PPCDFSA066MSFRBPHI",
                    "PRRESCONS",
                    "RPI",
                    "S4248SM144SCEN",
                    "SPCS20RSA",
                    "TCU",
                    "TEMPHELPS",
                    "TLHWYCONS",
                    "TLNRESCONS",
                    "TLRESCONS",
                    "USCONS",
                    "USFIRE",
                    "USGOOD",
                    "USPBS",
                    "USPHCI",
                    "USPRIV",
                    "USSLIND",
                    "USTPU",
                    "USTRADE",
                    "W825RC1",
                    "W875RX1",
                    "WHLSLRIRSA",
                    "A939RX0Q048SBEA",
                    "B009RC1Q027SBEA",
                    "CDSP",
                    "COFC",
                    "CP",
                    "CPATAX",
                    "FODSP",
                    "GDI",
                    "GDPC1",
                    "GGSAVE",
                    "GPDI",
                    "GPDIC1",
                    "GPSAVE",
                    "GSAVE",
                    "IEABC",
                    "IEAMGS",
                    "IEAXGS",
                    "MDSP",
                    "MMMFTAQ027S",
                    "NETFI",
                    "OUTMS",
                    "PNFI",
                    "PRFI",
                    "RSAHORUSQ156S",
                    "TDSP",
                    "ULCMFG",
                    "W207RC1Q156SBEA",
                    "W986RC1Q027SBEA",
                ],
                "COMMODITY": [
                    # "Aluminum",  # accuracy drops with this feature
                    "Baltic Dry",
                    "Brent",
                    "Coal",
                    "Cocoa",
                    # "Coffee",  # accuracy drops a lot with this feature
                    "Copper",
                    # "Corn",  # accuracy drops a lot with this feature
                    "Cotton",
                    "Crude Oil",
                    "Ethanol",
                    "Feeder Cattle",
                    "Gasoline",
                    "Gold",
                    "Heating Oil",
                    "Iron Ore",
                    "Lead",
                    "Lean Hogs",
                    "Live Cattle",
                    "LME Index",
                    "Lumber",
                    "Methanol",
                    "Natural gas",
                    "Propane",
                    "S&P GSCI",
                    "Silver",
                    "Soybeans",
                    "Sugar",
                    "Tin",
                    "Wheat",
                    "Zinc",
                ],
                "FI": [
                    # "Adj Close",
                    "T10Y2YM",
                    "T10YFFM",
                    "TB3SMFFM",
                    "T10YIEM",
                    "T5YIEM",
                    "GS10",
                    "GS30",
                    "GS2",
                    "TB3MS",
                ],
                "EQUITY": [
                    "Adj Close - Dow Jones",
                    "Adj Close - NASDAQ",
                    "Spx - Adj Close"
                ],
            }]
        ])


    def test_combo_bool(self, input_, expected):
        flattened_input_labels = [item for list_ in input_.values() for item in list_]
        output = compute_truth_table(flattened_input_labels)
        self.assertEqual(expected, output)
