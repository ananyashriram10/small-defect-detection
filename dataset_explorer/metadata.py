DATASET_ORDER = [
    "DAGM",
    "GC10-DET",
    "KolektorSDD2",
    "MPDD",
    "MTD",
    "Severstal",
    "VisA",
]

SIZE_ORDER = ["small", "medium", "large"]

SURFACE_TYPES = {
    "DAGM": "synthetic textured industrial surfaces",
    "GC10-DET": "steel/metal strip surfaces",
    "KolektorSDD2": "electrical commutator production surfaces",
    "MPDD": "painted metal/product surfaces",
    "MTD": "magnetic tile surfaces",
    "Severstal": "steel surface defects",
    "VisA": "mixed industrial object/surface categories",
}

LABEL_STRUCTURE = {
    "DAGM": "texture_class",
    "GC10-DET": "defect_type",
    "KolektorSDD2": "generic_defect",
    "MPDD": "product_type",
    "MTD": "defect_type",
    "Severstal": "generic_defect",
    "VisA": "product_type",
}

CLASS_TRANSLATIONS = {
    "DAGM": {
        "class1": "synthetic background texture class, generic surface defect",
        "class2": "synthetic background texture class, generic surface defect",
        "class3": "synthetic background texture class, generic surface defect",
        "class4": "synthetic background texture class, generic surface defect",
        "class5": "synthetic background texture class, generic surface defect",
        "class6": "synthetic background texture class, generic surface defect",
        "class7": "synthetic background texture class, generic surface defect",
        "class8": "synthetic background texture class, generic surface defect",
        "class9": "synthetic background texture class, generic surface defect",
        "class10": "synthetic background texture class, generic surface defect",
    },
    "GC10-DET": {
        "1_chongkong": "punching hole",
        "2_hanfeng": "weld line",
        "3_yueyawan": "crescent gap",
        "4_shuiban": "water spot",
        "5_youban": "oil spot",
        "6_siban": "silk spot",
        "7_yiwu": "inclusion",
        "8_yahen": "rolled pit",
        "9_zhehen": "crease",
        "10_yaozhe": "waist folding",
        "10_yaozhed": "waist folding",
        "d": "unclear, generic surface defect",
    },
    "KolektorSDD2": {
        "surface_defect": "generic surface defect",
    },
    "MPDD": {
        "bracket_black": "product type, generic surface anomaly",
        "bracket_brown": "product type, generic surface anomaly",
        "bracket_white": "product type, generic surface anomaly",
        "connector": "product type, generic surface anomaly",
        "metal_plate": "product type, generic surface anomaly",
        "tubes": "product type, generic surface anomaly",
    },
    "MTD": {
        "blowhole": "blowhole, gas cavity from casting",
        "break": "break, physical separation",
        "crack": "crack",
        "fray": "fray, worn or frayed edge",
        "free": "ambiguous, surface irregularity",
        "uneven": "uneven surface",
    },
    "Severstal": {
        "defect1": "generic surface defect, no official subtype meaning",
        "defect2": "generic surface defect, no official subtype meaning",
        "defect3": "generic surface defect, no official subtype meaning",
        "defect4": "generic surface defect, no official subtype meaning",
    },
    "VisA": {
        "candle": "product type, generic anomaly",
        "capsules": "product type, generic anomaly",
        "cashew": "product type, generic anomaly",
        "chewinggum": "product type, generic anomaly",
        "fryum": "product type, generic anomaly",
        "macaroni1": "product type, generic anomaly",
        "macaroni2": "product type, generic anomaly",
        "pcb1": "product type, generic anomaly",
        "pcb2": "product type, generic anomaly",
        "pcb3": "product type, generic anomaly",
        "pcb4": "product type, generic anomaly",
        "pipe_fryum": "product type, generic anomaly",
    },
}
