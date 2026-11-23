"""Köppen–Geiger classes (Beck et al. 2023): id -> (code, description, standard colour). From the notebook."""

KG_LEGEND = {
    1: ("Af", "Tropical rainforest", "#0000ff"), 2: ("Am", "Tropical monsoon", "#0078ff"),
    3: ("Aw", "Tropical savannah", "#46aafa"), 4: ("BWh", "Hot desert", "#ff0000"),
    5: ("BWk", "Cold desert", "#ff9696"), 6: ("BSh", "Hot steppe", "#f5a500"),
    7: ("BSk", "Cold steppe", "#ffdc64"), 8: ("Csa", "Mediterranean, hot summer", "#ffff00"),
    9: ("Csb", "Mediterranean, warm summer", "#c8c800"), 10: ("Csc", "Mediterranean, cold summer", "#969600"),
    11: ("Cwa", "Dry winter, hot summer", "#96ff96"), 12: ("Cwb", "Highland, dry winter", "#64c864"),
    13: ("Cwc", "Dry winter, cold summer", "#329632"), 14: ("Cfa", "Humid subtropical", "#c8ff50"),
    15: ("Cfb", "Oceanic", "#64ff50"), 16: ("Cfc", "Subpolar oceanic", "#32c800"),
    17: ("Dsa", "Continental, dry hot summer", "#ff00ff"), 18: ("Dsb", "Continental, dry warm summer", "#c800c8"),
    19: ("Dsc", "Continental, dry cool summer", "#963296"), 20: ("Dsd", "Continental, dry summer, very cold", "#966496"),
    21: ("Dwa", "Continental, dry winter, hot", "#aaafff"), 22: ("Dwb", "Continental, dry winter, warm", "#5a78dc"),
    23: ("Dwc", "Continental, dry winter, cool", "#4b50b4"), 24: ("Dwd", "Continental, dry winter, very cold", "#320087"),
    25: ("Dfa", "Continental, humid, hot", "#00ffff"), 26: ("Dfb", "Continental, humid, warm", "#37c8ff"),
    27: ("Dfc", "Subarctic", "#007d7d"), 28: ("Dfd", "Subarctic, very cold", "#00465f"),
    29: ("ET", "Tundra", "#b2b2b2"), 30: ("EF", "Ice cap", "#666666"),
}
