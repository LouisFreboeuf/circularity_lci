"""
Module implementing the full BAFU ecoSpold v1 → Brightway2 importer
exactly as used in the working notebook.

This module provides:

    import_bafu_from_sacchi(ecospold_folder, mapping_csv, db_name)

which performs:

    - reference index creation
    - full metadata + comment + representativeness parsing
    - extraction of main production flows
    - uncertainty field creation
    - biosphere mapping using CSV
    - Brightway LCIImporter integration

The module contains NO top‑level execution code.
It is meant to be called from your main notebook via setup_project().
"""

# =============================================================
# IMPORTS
# =============================================================

import bw2data as bd
import bw2io as bi

import logging

import numpy as np
import pandas as pd
import math
import csv
import ast
import json
import os
import html
import re

from pathlib import Path
from lxml import etree
from collections import defaultdict

from stats_arrays.distributions import (
    LognormalUncertainty,
    NormalUncertainty,
    TriangularUncertainty,
    UniformUncertainty,
    UndefinedUncertainty,
)

# =============================================================
# CONSTANTS
# =============================================================

logger = logging.getLogger(__name__)

NAME_LOC_RE = re.compile(r"^(.*)\s+\{([^{}]+)\}\s*$")

UNITS_MAP = {
    'kg': 'kilogram',
    'tkm': 'ton kilometer',
    'p': 'unit',
    'kWh': 'kilowatt hour',
    'MJ': 'megajoule',
    'm2': 'square meter',
    'm': 'meter',
    'Nm3': 'cubic meter',
    'km': 'kilometer',
    'personkm': 'person-kilometer',
    'my': 'meter-year',
    'unit': 'unit',
    'm3': 'cubic meter',
    'm2a': 'square meter-year',
    'kmy': 'kilometer-year',
    'a': 'year',
    'm3y': 'cubic meter-year',
    'kBq': 'kilo Becquerel',
    'ha': 'hectare',
    'Bq': 'Becquerel',
    'hr': 'hour'
}

CATS_MAP = {
    ('emissions to air', 'unspecified'): ('air',),
    ('emissions to air', 'high. pop.'): ('air', 'urban air close to ground'),
    ('emissions to air', 'low. pop.'): ('air', 'non-urban air or from high stacks'),
    ('emissions to air', 'stratosphere + troposphere'): ('air', 'lower stratosphere + upper troposphere'),
    ('emissions to air', 'low. pop., long-term'): ('air', 'low population density, long-term'),
    ('emissions to air', 'indoor'): ('air', 'urban air close to ground'),

    ('emissions to soil', 'unspecified'): ('soil',),
    ('emissions to soil', 'forestry'): ('soil', 'forestry'),
    ('emissions to soil', 'agricultural'): ('soil', 'agricultural'),
    ('emissions to soil', 'industrial'): ('soil', 'industrial'),

    ('emissions to water', 'ocean'): ('water', 'ocean'),
    ('emissions to water', 'river'): ('water', 'surface water'),
    ('emissions to water', 'unspecified'): ('water',),
    ('emissions to water', 'groundwater, long-term'): ('water', 'ground-, long-term'),
    ('emissions to water', 'groundwater'): ('water', 'ground-'),
    ('emissions to water', 'lake'): ('water', 'surface water'),
    ('emissions to water', 'river, long-term'): ('water', 'surface water'),
    ('emissions to water', 'fossilwater'): ('water', 'fossil well'),

    ('economic issues', 'unspecified'): ('economic', 'primary production factor'),

    ('resources', 'in ground'): ('natural resource', 'in ground'),
    ('resources', 'land'): ('natural resource', 'land'),
    ('resources', 'in water'): ('natural resource', 'in water'),
    ('resources', 'in air'): ('natural resource', 'in air'),
    ('resources', 'biotic'): ('natural resource', 'biotic'),
}

# =============================================================
# BASIC HELPERS
# =============================================================

def strip_location_from_name(name, current_location=None):
    if not name:
        return name, current_location

    m = NAME_LOC_RE.match(name)
    if not m:
        return name, current_location

    base, loc_str = m.groups()
    base = base.strip()
    loc = loc_str.strip()
    return base, (loc or current_location)


def get_first(elem, xpath):
    res = elem.xpath(xpath)
    return res[0] if res else None


def get_attr(elem, name, default=None):
    return elem.get(name) if elem is not None and elem.get(name) is not None else default


def person_label(person_elem):
    if person_elem is None:
        return None
    name = (get_attr(person_elem, "name", "") or "").strip()
    email = (get_attr(person_elem, "email", "") or "").strip()
    if name and email:
        return f"{name} ({email})"
    elif name:
        return name
    elif email:
        return email
    return None


def clean_comment(text):
    if not text:
        return ""
    t = html.unescape(text)
    t = re.sub(r"UUID:.*$", "", t, flags=re.S)
    lines = [ln.strip() for ln in t.replace("\r\n", "\n").split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)

# =============================================================
# UNCERTAINTY
# =============================================================

def add_uncertainty_fields_from_ecospold1(exc_xml, data):
    try:
        uncertainty = int(exc_xml.get("uncertaintyType", 0))
    except ValueError:
        uncertainty = 0

    def floatish(x):
        if x is None:
            return np.nan
        try:
            return float(x.strip())
        except Exception:
            try:
                return float(x)
            except Exception:
                return np.nan

    mean = floatish(exc_xml.get("meanValue"))
    min_ = floatish(exc_xml.get("minValue"))
    max_ = floatish(exc_xml.get("maxValue"))
    sigma95 = floatish(exc_xml.get("standardDeviation95"))

    if uncertainty == 1 and (sigma95 in (0, 1) or np.isnan(sigma95)):
        uncertainty = 0

    # LOGNORMAL
    if uncertainty == 1:
        if mean == 0 or np.isnan(mean):
            data.update({
                "uncertainty type": UndefinedUncertainty.id,
                "amount": float(mean),
                "loc": float(mean),
            })
            return data

        data.update({
            "uncertainty type": LognormalUncertainty.id,
            "amount": float(mean),
            "loc": math.log(abs(mean)),
            "scale": math.log(math.sqrt(float(sigma95))),
            "negative": mean < 0,
        })

        if np.isnan(data["scale"]):
            data["uncertainty type"] = UndefinedUncertainty.id
            data["loc"] = data["amount"]
            data.pop("scale", None)

        return data

    # NORMAL
    if uncertainty == 2:
        data.update({
            "uncertainty type": NormalUncertainty.id,
            "amount": float(mean),
            "loc": float(mean),
            "scale": float(sigma95) / 2.0,
        })
        return data

    # TRIANGULAR
    if uncertainty == 3:
        most_likely = floatish(exc_xml.get("mostLikelyValue"))
        data.update({
            "uncertainty type": TriangularUncertainty.id,
            "minimum": float(min_),
            "maximum": float(max_),
        })
        data["amount"] = data["loc"] = (
            most_likely if not np.isnan(most_likely) else float(mean)
        )
        return data

    # UNIFORM
    if uncertainty == 4:
        data.update({
            "uncertainty type": UniformUncertainty.id,
            "amount": float(mean),
            "minimum": float(min_),
            "maximum": float(max_),
        })
        return data

    # UNDEFINED
    data.update({
        "uncertainty type": UndefinedUncertainty.id,
        "amount": float(mean),
        "loc": float(mean),
    })
    return data

# =============================================================
# REFERENCE INDEX
# =============================================================

def build_reference_index(root_dir: Path):
    """
    First pass: retrieve main production exchanges (outputGroup==0)
    to map flow numbers to activity reference products.
    """
    ref_index = {}

    for xml in root_dir.rglob("*.xml"):
        tree = etree.parse(str(xml))
        root = tree.getroot()

        dataset = get_first(root, '//*[local-name()="dataset"]')
        if dataset is None:
            continue

        PI = get_first(dataset, './/*[local-name()="processInformation"]')
        RF = get_first(PI, './/*[local-name()="referenceFunction"]') if PI else None
        geo = get_first(PI, './/*[local-name()="geography"]') if PI else None

        if RF is None:
            continue

        raw_name = get_attr(RF, "name", "") or ""
        unit = get_attr(RF, "unit", "")
        xml_location = get_attr(geo, "location", None) or "GLO"
        dataset_number = get_attr(dataset, "number", "")

        clean_name, location = strip_location_from_name(raw_name, xml_location)

        prod_flow_id = None
        flow_data = get_first(dataset, './/*[local-name()="flowData"]')

        if flow_data is not None:
            for exc in flow_data.xpath('./*[local-name()="exchange"]'):
                og = get_first(exc, './*[local-name()="outputGroup"]')
                og_val = og.text.strip() if og is not None and og.text else None
                if og_val == "0":
                    prod_flow_id = get_attr(exc, "number", None)
                    if prod_flow_id:
                        break

        flow_id = prod_flow_id or dataset_number
        if not flow_id:
            continue

        ref_index[flow_id] = {
            "activity_code": dataset_number,
            "name": clean_name,
            "reference product": clean_name,
            "location": location,
            "unit": unit,
        }

    return ref_index

# =============================================================
# PARSE CATEGORIES
# =============================================================

def safe_parse_category(cat_str):
    if cat_str is None:
        return None
    cat_str = cat_str.strip()
    if cat_str == "":
        return None

    if cat_str.startswith("(") or cat_str.startswith("["):
        try:
            return tuple(ast.literal_eval(cat_str))
        except Exception:
            return (cat_str,)

    return (cat_str,)

# =============================================================
# PARSE ECOSPOLD FILE
# =============================================================

def parse_ecospold_file(xml_path: Path, db_name: str, ref_index: dict):

    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    dataset = get_first(root, '//*[local-name()="dataset"]')
    if dataset is None:
        return None

    dataset_number = get_attr(dataset, "number", "")

    PI = get_first(dataset, './/*[local-name()="processInformation"]')
    RF = get_first(PI, './/*[local-name()="referenceFunction"]') if PI else None
    geo = get_first(PI, './/*[local-name()="geography"]') if PI else None
    TP = get_first(PI, './/*[local-name()="timePeriod"]') if PI else None

    if RF is None:
        return None

    raw_name = get_attr(RF, "name", "") or ""
    unit = UNITS_MAP.get(get_attr(RF, "unit", ""))
    xml_location = get_attr(geo, "location", None) or "GLO"

    name, location = strip_location_from_name(raw_name, xml_location)
    ref_product = name

    # ---------------------------------------------------------
    # COMMENT BLOCK
    # ---------------------------------------------------------
    rf_comment = get_attr(RF, "generalComment", "") or ""

    tp_text = get_attr(TP, "text", "") or ""
    tp_start_elem = get_first(TP, './/*[local-name()="startDate"]') if TP else None
    tp_end_elem = get_first(TP, './/*[local-name()="endDate"]') if TP else None
    tp_start = (tp_start_elem.text or "").strip() if tp_start_elem is not None else ""
    tp_end = (tp_end_elem.text or "").strip() if tp_end_elem is not None else ""

    time_lines = []
    if tp_text:
        time_lines.append(f"Time period: {tp_text}")
    if tp_start or tp_end:
        if tp_start and tp_end:
            time_lines.append(f"Time period (data): {tp_start} – {tp_end}")
        elif tp_start:
            time_lines.append(f"Time period (data): from {tp_start}")
        elif tp_end:
            time_lines.append(f"Time period (data): until {tp_end}")

    geo_text = get_attr(geo, "text", "") or ""
    geo_lines = []
    if xml_location or geo_text:
        if geo_text:
            geo_lines.append(f"Geography: {xml_location} – {geo_text}")
        else:
            geo_lines.append(f"Geography: {xml_location}")

    tech = get_first(PI, './/*[local-name()="technology"]') if PI else None
    tech_text = get_attr(tech, "text", "") or ""
    tech_lines = [f"Technology: {tech_text}"] if tech_text else []

    MI = get_first(dataset, './/*[local-name()="metaInformation"]')
    AI = get_first(MI, './/*[local-name()="administrativeInformation"]') if MI else None
    MV = get_first(MI, './/*[local-name()="modellingAndValidation"]') if MI else None

    meta_lines = []

    # REPRESENTATIVENESS
    rep = get_first(MV, './/*[local-name()="representativeness"]') if MV else None
    if rep is not None:
        rep_parts = []

        prod_vol = get_attr(rep, "productionVolume", "") or ""
        sampling = get_attr(rep, "samplingProcedure", "") or ""
        extrap = get_attr(rep, "extrapolations", "") or ""
        unc_adj = get_attr(rep, "uncertaintyAdjustments", "") or ""

        if prod_vol and prod_vol.lower() != "na":
            rep_parts.append(f"Production volume: {prod_vol}")
        if sampling and sampling.lower() != "<null>":
            rep_parts.append(f"Sampling: {sampling}")
        if extrap and extrap.lower() != "<null>":
            rep_parts.append(f"Extrapolations: {extrap}")
        if unc_adj and unc_adj.lower() != "none":
            rep_parts.append(f"Uncertainty adjustments: {unc_adj}")

        if rep_parts:
            meta_lines.append("Representativeness: " + "; ".join(rep_parts))

    # PEOPLE
    persons = {}
    if AI is not None:
        for p in AI.xpath('.//*[local-name()="person"]'):
            num = get_attr(p, "number", None)
            if num:
                persons[num] = p

        de = get_first(AI, './/*[local-name()="dataEntryBy"]')
        de_num = get_attr(de, "person", None) if de else None
        de_label = person_label(persons.get(de_num))
        if de_label:
            meta_lines.append(f"Data entry: {de_label}")

        dgp = get_first(AI, './/*[local-name()="dataGeneratorAndPublication"]')
        dgp_num = get_attr(dgp, "person", None) if dgp else None
        dgp_label = person_label(persons.get(dgp_num))
        if dgp_label:
            meta_lines.append(f"Data generator: {dgp_label}")

    # VALIDATION
    if MV is not None:
        val = get_first(MV, './/*[local-name()="validation"]')
        if val is not None:
            details = get_attr(val, "proofReadingDetails", "") or ""
            validator_num = get_attr(val, "proofReadingValidator", None)
            validator_label = person_label(persons.get(validator_num)) if validator_num else None

            if validator_label:
                meta_lines.append(f"Proof-reading validator: {validator_label}")
            if details:
                meta_lines.append(f"Validation details: {details}")

    # SOURCE / PUBLICATION INFO
    if MV is not None:
        src = get_first(MV, './/*[local-name()="source"]')
        if src is not None:
            first_author = get_attr(src, "firstAuthor", "") or ""
            add_authors = get_attr(src, "additionalAuthors", "") or ""
            year = get_attr(src, "year", "") or ""
            title = get_attr(src, "title", "") or ""
            publisher = get_attr(src, "publisher", "") or ""
            place = get_attr(src, "placeOfPublications", "") or ""
            volume = get_attr(src, "volumeNo", "") or ""

            src_parts = []
            if first_author:
                src_parts.append(first_author)
            if add_authors:
                src_parts.append(add_authors)
            if year:
                src_parts.append(f"({year})")
            if title:
                src_parts.append(title)
            if publisher or place or volume:
                pub_bits = ", ".join(
                    [x for x in [publisher, place, f"vol. {volume}" if volume else ""] if x]
                )
                if pub_bits:
                    src_parts.append(pub_bits)

            if src_parts:
                meta_lines.append("Source: " + " ".join(src_parts))

    comment_sections = []
    if rf_comment:
        comment_sections.append(rf_comment)
    comment_sections.extend(time_lines)
    comment_sections.extend(geo_lines)
    comment_sections.extend(tech_lines)
    comment_sections.extend(meta_lines)

    comment = clean_comment("\n".join(cs for cs in comment_sections if cs))

    # CLASSIFICATIONS
    category = get_attr(RF, "category", None)
    subcategory = get_attr(RF, "subCategory", None)
    classifications = []
    if category or subcategory:
        classifications.append(("EcoSpold01Categories", f"{category or ''}/{subcategory or ''}"))

    # ---------------------------------------------------------
    # EXCHANGES
    # ---------------------------------------------------------

    flow_data = get_first(dataset, './/*[local-name()="flowData"]')
    exchanges = []

    if flow_data is not None:
        for exc in flow_data.xpath('./*[local-name()="exchange"]'):
            mean_value = get_attr(exc, "meanValue", None)
            if mean_value is None:
                continue

            raw_ex_name = get_attr(exc, "name", "") or ""
            ex_unit = UNITS_MAP.get(get_attr(exc, "unit", ""))
            ex_cat = get_attr(exc, "category", "")
            ex_subcat = get_attr(exc, "subCategory", "")
            ex_loc_xml = get_attr(exc, "location", None)

            ex_name, ex_loc_from_name = strip_location_from_name(raw_ex_name, ex_loc_xml)
            ex_loc = ex_loc_from_name

            og = get_first(exc, './*[local-name()="outputGroup"]')
            ig = get_first(exc, './*[local-name()="inputGroup"]')
            og_val = og.text.strip() if og is not None and og.text else None
            ig_val = ig.text.strip() if ig is not None and ig.text else None

            ex_cat_lower = (ex_cat or "").lower()
            ex_subcat_lower = (ex_subcat or "").lower()

            is_biosphere_category = (
                ex_cat_lower.startswith("emissions to ")
                or ex_cat_lower.startswith("emission to ")
                or ex_cat_lower in {"emissions", "emission"}
                or "resource" in ex_cat_lower
                or (ex_cat, ex_subcat) in CATS_MAP
            )

            if og_val == "0":
                ex_type = "production"
            elif is_biosphere_category:
                ex_type = "biosphere"
            elif ig_val is not None:
                ex_type = "technosphere"
            else:
                ex_type = "biosphere"

            exc_dict = {
                "name": ex_name,
                "unit": ex_unit,
                "type": ex_type,
                "categories": CATS_MAP.get((ex_cat, ex_subcat), (ex_cat, ex_subcat)),
            }

            if ex_loc:
                exc_dict["location"] = ex_loc

            ex_comment = exc.get("generalComment")
            if ex_comment:
                exc_dict["comment"] = clean_comment(ex_comment)

            ex_number = get_attr(exc, "number", None)
            if ex_number:
                exc_dict["flow"] = ex_number

                if ex_type in ("technosphere", "production"):
                    ref_info = ref_index.get(ex_number)
                    if ref_info:
                        exc_dict["reference product"] = ref_info["reference product"]
                        if "location" not in exc_dict or not exc_dict["location"]:
                            exc_dict["location"] = ref_info["location"]

            add_uncertainty_fields_from_ecospold1(exc, exc_dict)

            exchanges.append(exc_dict)

    # ---------------------------------------------------------
    # COMPLETE DATASET
    # ---------------------------------------------------------

    data = {
        "database": db_name,
        "code": dataset_number,
        "name": name,
        "reference product": ref_product,
        "location": location,
        "unit": unit,
        "comment": comment,
        "classifications": classifications,
        "exchanges": exchanges,
        "filename": xml_path.name,
        "type": "process",
    }

    return data

# =============================================================
# MAIN IMPORT FUNCTION
# =============================================================

def import_bafu_from_sacchi(ecospold_folder, mapping_csv=None, db_name="bafu_import"):
    """
    Full import procedure callable from your main notebook.

    If ``mapping_csv`` is not provided, the BAFU elementary flows mapping
    shipped with the package (``circularity_lci/data/elementary_flows_mapping.csv``)
    is used, resolved independently of the current working directory.
    """

    from ._packaging import get_data_path

    ecospold_folder = Path(ecospold_folder)
    if mapping_csv is None:
        mapping_csv = get_data_path("elementary_flows_mapping.csv")
    mapping_csv = Path(mapping_csv)

    logger.info("Creating biosphere3 if needed...")
    if "biosphere3" not in bd.databases:
        bi.create_default_biosphere3()

    logger.info("Building reference index...")
    ref_index = build_reference_index(ecospold_folder)
    logger.info("Reference index contains %d entries.", len(ref_index))

    logger.info("Parsing ecoSpold files...")
    all_data = []
    for xml in ecospold_folder.rglob("*.xml"):
        dct = parse_ecospold_file(xml, db_name, ref_index)
        if dct is not None:
            all_data.append(dct)

    logger.info("Built %d dataset dictionaries.", len(all_data))

    # ---------------------------------------------------------
    # BIOSPHERE MAPPING
    # ---------------------------------------------------------
    logger.info("Loading biosphere mapping...")
    mapping = {}

    with open(mapping_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bafu_name = row["BAFU name"].strip()
            bafu_cat = safe_parse_category(row["BAFU category"])

            key = (bafu_name, bafu_cat)

            mapping[key] = {
                "ecoinvent_name": (row.get("Ecoinvent name") or "").strip() or None,
                "ecoinvent_category": safe_parse_category(row.get("Ecoinvent category")),
            }

    logger.info("%d mapping entries loaded.", len(mapping))

    logger.info("Applying biosphere mapping...")
    for ds in all_data:
        for e in ds["exchanges"]:
            if e["type"] == "biosphere":
                key = (e["name"], e["categories"])
                if key in mapping:
                    new_name = mapping[key]["ecoinvent_name"]
                    new_cat = mapping[key]["ecoinvent_category"]

                    if new_name:
                        e["name"] = new_name
                    if new_cat:
                        e["categories"] = new_cat

    # ---------------------------------------------------------
    # BRIGHTWAY IMPORT
    # ---------------------------------------------------------
    logger.info("Importing database into Brightway...")

    importer = bi.importers.base_lci.LCIImporter(db_name=db_name)
    importer.data = all_data

    importer.apply_strategies()
    importer.match_database(fields=["name", "reference product", "location"])
    importer.match_database("biosphere3", fields=["name", "categories"])

    importer.drop_unlinked(i_am_reckless=True)
    importer.write_database()

    logger.info("Database '%s' successfully imported.", db_name)
