import os
import time
import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import bw2data as bd
import bw2calc as bc

from .multi_lca_calculator import MultiLCACalculator
from .circularity_calculator import CircularityCalculator


logger = logging.getLogger(__name__)


class CircularityDatabaseAnalyzer:
    """
    Compute circularity indicators for all activities in a Brightway database.

    Architecture
    ------------
    CircularityCalculator is the single source of truth for:
        - inventory retrieval
        - flow exclusion
        - mass factors
        - energy factors
        - flow classification
        - circularity equations
        - LFI / CFI / efficiency calculations

    CircularityDatabaseAnalyzer is responsible for:
        - selecting database activities
        - constructing the functional unit for each activity
        - calling CircularityCalculator
        - adding activity/database metadata
        - batch processing
        - interim/final CSV output
        - ISIC and location classification

    This design ensures that a circularity calculation performed for a
    single functional unit uses exactly the same methodology as the
    corresponding calculation performed through the database analyzer.
    """

    def __init__(
        self,
        project_name,
        isic_section_map,
        isic_division_map,
        location_to_un_group,
        excluded_flows,
        valuable_water_compartments,
        exclude_water,
        technosphere_db_name,
        biosphere_db_name,
        mass_strategy,
        energy_strategy,
    ):
        self.project_name = project_name

        self.EXCLUDED_FLOWS = excluded_flows
        self.VALUABLE_WATER_COMPARTMENTS = valuable_water_compartments
        self.EXCLUDE_WATER = exclude_water

        self.ISIC_SECTION_MAP = isic_section_map
        self.ISIC_DIVISION_MAP = isic_division_map
        self.LOCATION_TO_UN_GROUP = location_to_un_group

        self.technosphere_db_name = technosphere_db_name
        self.biosphere_db_name = biosphere_db_name

        self.mass_strategy = mass_strategy
        self.energy_strategy = energy_strategy

        # Kept from the original module for compatibility with existing code.
        self.multi_lca_calculator = MultiLCACalculator(
            technosphere_db_name
        )

        # IMPORTANT:
        # CircularityCalculator is the authoritative circularity engine.
        self.circularity_calculator = CircularityCalculator(
            project_name,
            excluded_flows,
            valuable_water_compartments,
            exclude_water=exclude_water,
            technosphere_db_name=technosphere_db_name,
            biosphere_db_name=biosphere_db_name,
            mass_strategy=mass_strategy,
            energy_strategy=energy_strategy,
        )

    # ------------------------------------------------------------------
    # ISIC / LOCATION CLASSIFICATION
    # ------------------------------------------------------------------

    def get_isic_class(self, activity):
        """
        Retrieve ISIC class from Brightway activity metadata.
        """
        isic_class = (
            activity.get("ISIC")
            or activity.get("ISIC code")
            or "Unknown"
        )

        if isic_class == "Unknown":
            classifications = activity.get("classifications", [])

            for classification in classifications:
                if (
                    isinstance(classification, tuple)
                    and len(classification) >= 2
                    and classification[0] == "ISIC rev.4 ecoinvent"
                ):
                    isic_class = classification[1]
                    break

        return isic_class

    def parse_isic(self, isic_code):
        """
        Parse ISIC class into section, division, and description.
        """
        try:
            if not isic_code or isic_code == "Unknown":
                raise ValueError("Unknown ISIC code")

            isic_code = str(isic_code)
            division = isic_code[:2]

            section, description = self.ISIC_DIVISION_MAP.get(
                division,
                ("Unknown", "Unknown"),
            )

            return {
                "division": division,
                "section": section,
                "description": description,
            }

        except Exception:
            return {
                "division": "Unknown",
                "section": "Unknown",
                "description": "Unknown",
            }

    def precompute_isic_classes(self):
        """
        Precompute ISIC classes for all activities in the database.
        """
        db = bd.Database(self.technosphere_db_name)

        isic_lookup = {}

        for activity in db:
            isic_class = self.get_isic_class(activity)
            isic_lookup[activity.key] = isic_class

        return isic_lookup

    def precompute_isic_classes_classified(self):
        """
        Precompute ISIC classes for all activities in the database,
        including section and division.
        """
        db = bd.Database(self.technosphere_db_name)

        isic_lookup = {}

        for activity in db:
            isic_class = None

            for classification in activity.get("classifications", []):
                if (
                    isinstance(classification, tuple)
                    and len(classification) >= 2
                    and str(classification[0]).startswith("ISIC")
                ):
                    isic_class = str(classification[1]).split(":")[0]
                    break

            if isic_class:
                parsed = self.parse_isic(isic_class)

                isic_lookup[activity.key] = {
                    "isic_code": isic_class,
                    "division": parsed["division"],
                    "section": parsed["section"],
                    "description": parsed["description"],
                }

            else:
                isic_lookup[activity.key] = {
                    "isic_code": "Unknown",
                    "division": "Unknown",
                    "section": "Unknown",
                    "description": "Unknown",
                }

        return isic_lookup

    def precompute_locations_classified(self):
        """
        Precompute locations for all activities in the database,
        classified by UN Regional Group.
        """
        db = bd.Database(self.technosphere_db_name)

        location_lookup_classified = {}

        for activity in db:
            location = activity.get("location", "Unknown")

            un_group = self.LOCATION_TO_UN_GROUP.get(
                location,
                "Unknown",
            )

            location_lookup_classified[activity.key] = {
                "location": location,
                "un_group": un_group,
            }

        return location_lookup_classified

    # ------------------------------------------------------------------
    # INVENTORY
    # ------------------------------------------------------------------

    def get_inventory_for_1_act(self, fu_dict):
        """
        Get inventory flows for a single functional unit.

        IMPORTANT
        ---------
        Inventory retrieval is delegated to CircularityCalculator so that
        the database analyzer and the standalone calculator use exactly
        the same inventory definition.

        Unlike the original database implementation, this method does NOT
        manually add the reference product to the inventory.
        """
        try:
            activity_key = fu_dict["activity_key"]
            amount = fu_dict["amount"]

            functional_unit = {
                activity_key: amount
            }

            flows_df = self.circularity_calculator.get_inventory_flows(
                functional_unit=functional_unit
            )

            return flows_df

        except Exception as e:
            logger.exception(
                "LCA error for activity %s: %s",
                fu_dict.get("activity_key"),
                e,
            )

            return pd.DataFrame()

    # ------------------------------------------------------------------
    # CIRCULARITY RESULT EXTRACTION
    # ------------------------------------------------------------------

    @staticmethod
    def _get_result_row(combined_df, unit):
        """
        Safely retrieve one unit row from the CircularityCalculator
        summary DataFrame.
        """
        if combined_df is None or combined_df.empty:
            return None

        if unit not in combined_df.index:
            return None

        return combined_df.loc[unit]

    @staticmethod
    def _safe_value(row, column, default=0.0):
        """
        Safely extract a numeric value from a pandas row.
        """
        if row is None:
            return default

        try:
            value = row[column]

            if pd.isna(value):
                return default

            return float(value)

        except (KeyError, TypeError, ValueError):
            return default

    def _extract_circularity_metrics(
        self,
        combined_df,
        inefficiency,
        efficiency,
    ):
        """
        Extract database-level mass and energy metrics from the authoritative
        CircularityCalculator result.

        The selected mass strategy is represented by the 'kilogram' row.
        The selected energy strategy is represented by the 'megajoule' row.

        Returns
        -------
        dict
            Metrics using the original database analyzer output names.
        """

        mass_row = self._get_result_row(
            combined_df,
            "kilogram",
        )

        energy_row = self._get_result_row(
            combined_df,
            "megajoule",
        )

        # --------------------------------------------------------------
        # Mass metrics
        # --------------------------------------------------------------

        V_kg = self._safe_value(
            mass_row,
            "Natural Resources",
        )

        Ri_kg = self._safe_value(
            mass_row,
            "Technosphere Inputs",
        )

        Rr_kg = self._safe_value(
            mass_row,
            "Recycled Outputs",
        )

        Er_kg = self._safe_value(
            mass_row,
            "Cleaned Emissions",
        )

        W_kg = self._safe_value(
            mass_row,
            "Waste Emissions",
        )

        mass_denominator = V_kg + Ri_kg

        eta_minus_kg = (
            W_kg / mass_denominator
            if mass_denominator
            else 0.0
        )

        eta_plus_kg = (
            (Rr_kg + Er_kg) / mass_denominator
            if mass_denominator
            else 0.0
        )

        LFI_kg = (
            (W_kg + V_kg)
            / (2 * mass_denominator)
            if mass_denominator
            else 0.0
        )

        CFI_kg = (
            (Rr_kg + Er_kg + Ri_kg)
            / (2 * mass_denominator)
            if mass_denominator
            else 0.0
        )

        # --------------------------------------------------------------
        # Energy metrics
        # --------------------------------------------------------------

        V_MJ = self._safe_value(
            energy_row,
            "Natural Resources",
        )

        Ri_MJ = self._safe_value(
            energy_row,
            "Technosphere Inputs",
        )

        Rr_MJ = self._safe_value(
            energy_row,
            "Recycled Outputs",
        )

        Er_MJ = self._safe_value(
            energy_row,
            "Cleaned Emissions",
        )

        W_MJ = self._safe_value(
            energy_row,
            "Waste Emissions",
        )

        energy_denominator = V_MJ + Ri_MJ

        eta_minus_MJ = (
            W_MJ / energy_denominator
            if energy_denominator
            else 0.0
        )

        eta_plus_MJ = (
            (Rr_MJ + Er_MJ) / energy_denominator
            if energy_denominator
            else 0.0
        )

        LFI_MJ = (
            (W_MJ + V_MJ)
            / (2 * energy_denominator)
            if energy_denominator
            else 0.0
        )

        CFI_MJ = (
            (Rr_MJ + Er_MJ + Ri_MJ)
            / (2 * energy_denominator)
            if energy_denominator
            else 0.0
        )

        # --------------------------------------------------------------
        # Use the authoritative efficiency series where available.
        #
        # The formulas above reproduce the existing database output
        # structure, while these values come directly from the common
        # CircularityCalculator.
        # --------------------------------------------------------------

        eta_minus_kg_authoritative = 0.0
        eta_plus_kg_authoritative = 0.0
        eta_minus_MJ_authoritative = 0.0
        eta_plus_MJ_authoritative = 0.0

        if inefficiency is not None:
            try:
                if "kilogram" in inefficiency.index:
                    eta_minus_kg_authoritative = float(
                        inefficiency.loc["kilogram"]
                    )
            except (KeyError, TypeError, ValueError):
                pass

            try:
                if "megajoule" in inefficiency.index:
                    eta_minus_MJ_authoritative = float(
                        inefficiency.loc["megajoule"]
                    )
            except (KeyError, TypeError, ValueError):
                pass

        if efficiency is not None:
            try:
                if "kilogram" in efficiency.index:
                    eta_plus_kg_authoritative = float(
                        efficiency.loc["kilogram"]
                    )
            except (KeyError, TypeError, ValueError):
                pass

            try:
                if "megajoule" in efficiency.index:
                    eta_plus_MJ_authoritative = float(
                        efficiency.loc["megajoule"]
                    )
            except (KeyError, TypeError, ValueError):
                pass

        # Prefer the values directly returned by the authoritative
        # calculator whenever those rows exist.
        if mass_row is not None:
            eta_minus_kg = eta_minus_kg_authoritative
            eta_plus_kg = eta_plus_kg_authoritative

        if energy_row is not None:
            eta_minus_MJ = eta_minus_MJ_authoritative
            eta_plus_MJ = eta_plus_MJ_authoritative

        return {
            "V_kg": V_kg,
            "Ri_kg": Ri_kg,
            "Rr_kg": Rr_kg,
            "Er_kg": Er_kg,
            "W_kg": W_kg,
            "eta-_kg": eta_minus_kg,
            "eta+_kg": eta_plus_kg,
            "LFI_kg": LFI_kg,
            "CFI_kg": CFI_kg,

            "V_MJ": V_MJ,
            "Ri_MJ": Ri_MJ,
            "Rr_MJ": Rr_MJ,
            "Er_MJ": Er_MJ,
            "W_MJ": W_MJ,
            "eta-_MJ": eta_minus_MJ,
            "eta+_MJ": eta_plus_MJ,
            "LFI_MJ": LFI_MJ,
            "CFI_MJ": CFI_MJ,
        }

    # ------------------------------------------------------------------
    # SINGLE FUNCTIONAL UNIT
    # ------------------------------------------------------------------

    def compute_circularity_for_single_fu(
        self,
        fu_dict,
        mass_lookup=None,
        energy_lookup=None,
        water_lookup=None,
        dry_mass_lookup=None,
        ced_total_dict=None,
        ced_renewable_dict=None,
        ced_nonrenewable_dict=None,
        isic_lookup_classified=None,
        location_lookup_classified=None,
    ):
        """
        Compute circularity for a single functional unit.

        Parameters
        ----------
        fu_dict : dict
            Expected structure:

                {
                    "activity_key": activity.key,
                    "amount": production_exchange_amount,
                    "activity": activity
                }

        The lookup arguments are retained in the method signature for
        backward compatibility with the original module.

        They are intentionally NOT used for circularity calculations.

        CircularityCalculator now owns:
            - lookup construction
            - flow classification
            - mass/energy conversion
            - circularity calculations

        This eliminates the duplicate implementation previously present
        in this method.
        """

        try:
            activity_key = fu_dict["activity_key"]
            amount = fu_dict["amount"]

            # ----------------------------------------------------------
            # Retrieve activity
            # ----------------------------------------------------------

            activity = fu_dict.get("activity")

            if activity is None:
                activity = bd.get_activity(activity_key)

            if activity is None:
                logger.warning(
                    "Activity %s could not be retrieved.",
                    activity_key,
                )
                return None

            # ----------------------------------------------------------
            # Construct the SAME direct functional unit that is passed
            # to CircularityCalculator in standalone calculations.
            # ----------------------------------------------------------

            functional_unit = {
                activity_key: amount
            }

            # ----------------------------------------------------------
            # AUTHORITATIVE CIRCULARITY CALCULATION
            #
            # IMPORTANT:
            # compute_circularity_efficiency_variables returns:
            #
            #   combined_df,
            #   inefficiency,
            #   efficiency,
            #   detailed_flows_df
            #
            # The previous database implementation incorrectly treated
            # this return value as a DataFrame/dict, producing:
            #
            #   Unexpected result type ... <class 'tuple'>
            # ----------------------------------------------------------

            (
                combined_df,
                inefficiency,
                efficiency,
                detailed_flows_df,
            ) = self.circularity_calculator.compute_circularity_efficiency_variables(
                functional_unit=functional_unit,
                mass_strategy=self.mass_strategy,
                energy_strategy=self.energy_strategy,
                save_csv=False,
            )

            if combined_df is None or combined_df.empty:
                logger.warning(
                    "No circularity result for activity %s",
                    activity_key,
                )
                return None

            # ----------------------------------------------------------
            # Extract the metrics from the authoritative result.
            # ----------------------------------------------------------

            metrics = self._extract_circularity_metrics(
                combined_df,
                inefficiency,
                efficiency,
            )

            # ----------------------------------------------------------
            # Metadata
            # ----------------------------------------------------------

            if isic_lookup_classified is None:
                isic_lookup_classified = {}

            if location_lookup_classified is None:
                location_lookup_classified = {}

            isic_info = isic_lookup_classified.get(
                activity.key,
                {
                    "isic_code": "Unknown",
                    "division": "Unknown",
                    "section": "Unknown",
                    "description": "Unknown",
                },
            )

            location_info = location_lookup_classified.get(
                activity.key,
                {},
            )

            location_code = location_info.get(
                "location",
                activity.get("location", "Unknown"),
            )

            un_group = location_info.get(
                "un_group",
                "Unknown",
            )

            # ----------------------------------------------------------
            # Preserve the original database analyzer output structure.
            # ----------------------------------------------------------

            return {
                "Process Key": activity.key,
                "Process Name": activity.get("name"),
                "Location": activity.get("location", "Unknown"),
                "Reference Product": activity.get(
                    "reference product",
                    "Unknown",
                ),
                "Unit": activity.get("unit", "Unknown"),

                "ISIC Code": isic_info["isic_code"],
                "ISIC Division": isic_info["division"],
                "ISIC Section": isic_info["section"],
                "ISIC Description": isic_info["description"],

                "LOCATION Code": location_code,
                "UN Group": un_group,

                "V_kg": metrics["V_kg"],
                "Ri_kg": metrics["Ri_kg"],
                "Rr_kg": metrics["Rr_kg"],
                "Er_kg": metrics["Er_kg"],
                "W_kg": metrics["W_kg"],
                "eta-_kg": metrics["eta-_kg"],
                "eta+_kg": metrics["eta+_kg"],
                "LFI_kg": metrics["LFI_kg"],
                "CFI_kg": metrics["CFI_kg"],

                "V_MJ": metrics["V_MJ"],
                "Ri_MJ": metrics["Ri_MJ"],
                "Rr_MJ": metrics["Rr_MJ"],
                "Er_MJ": metrics["Er_MJ"],
                "W_MJ": metrics["W_MJ"],
                "eta-_MJ": metrics["eta-_MJ"],
                "eta+_MJ": metrics["eta+_MJ"],
                "LFI_MJ": metrics["LFI_MJ"],
                "CFI_MJ": metrics["CFI_MJ"],
            }

        except Exception as e:
            logger.exception(
                "Error processing activity %s",
                fu_dict.get("activity_key"),
            )
            return None

    # ------------------------------------------------------------------
    # DATABASE-WIDE SEQUENTIAL CALCULATION
    # ------------------------------------------------------------------

    def compute_circularity_for_all_processes_sequential(
        self,
        sample_size=None,
        save_interval=500,
    ):
        """
        Compute circularity metrics for all processes sequentially.

        The functional unit for each activity is defined from its first
        production exchange, preserving the original database analyzer
        behavior.

        Circularity itself is delegated completely to
        CircularityCalculator.
        """

        db = bd.Database(self.technosphere_db_name)

        activities = list(db)

        logger.info(
            "Total activities in '%s': %d",
            self.technosphere_db_name,
            len(activities),
        )

        if not activities:
            logger.warning(
                "No activities found in the database."
            )
            return pd.DataFrame()

        # --------------------------------------------------------------
        # Optional deterministic sampling
        # --------------------------------------------------------------

        if sample_size and sample_size < len(activities):
            activities = (
                pd.Series(activities)
                .sample(
                    sample_size,
                    random_state=42,
                )
                .tolist()
            )

            logger.info(
                "Sampling %d activities using random_state=42",
                len(activities),
            )

        # --------------------------------------------------------------
        # Build functional units
        # --------------------------------------------------------------

        functional_units = []

        for activity in activities:
            try:
                production_exchanges = [
                    exc
                    for exc in activity.production()
                    if exc.get("type") == "production"
                ]

                if production_exchanges:
                    production_exc = production_exchanges[0]

                    functional_units.append(
                        {
                            "activity_key": activity.key,
                            "amount": production_exc["amount"],
                            "activity": activity,
                        }
                    )

            except Exception:
                logger.exception(
                    "Could not obtain production exchange for %s",
                    activity.key,
                )

        if not functional_units:
            logger.warning(
                "No functional units prepared. "
                "Check if activities have production exchanges."
            )
            return pd.DataFrame()

        logger.info(
            "Prepared %d functional units",
            len(functional_units),
        )

        # --------------------------------------------------------------
        # Metadata lookups only.
        #
        # Circularity lookups are deliberately NOT built here anymore.
        # CircularityCalculator handles them internally.
        # --------------------------------------------------------------

        isic_lookup_classified = (
            self.precompute_isic_classes_classified()
        )

        location_lookup_classified = (
            self.precompute_locations_classified()
        )

        results = []

        start_time = time.time()

        logger.info(
            "Starting sequential processing of %d processes...",
            len(functional_units),
        )

        # --------------------------------------------------------------
        # Process each functional unit
        # --------------------------------------------------------------

        for i, fu_dict in enumerate(functional_units):

            try:
                if i % 10 == 0 or i == len(functional_units) - 1:
                    elapsed = time.time() - start_time

                    rate = (
                        (i + 1) / elapsed
                        if elapsed > 0
                        else 0
                    )

                    remaining = (
                        (len(functional_units) - i - 1) / rate
                        if rate > 0
                        else float("inf")
                    )

                    logger.info(
                        "Processing %d/%d (%.1f%%) - Rate: %.1f processes/min - ETA: %.1f minutes",
                        i + 1,
                        len(functional_units),
                        ((i + 1) / len(functional_units)) * 100,
                        rate * 60,
                        remaining / 60,
                    )

                # ------------------------------------------------------
                # Circularity calculation is delegated to the common
                # CircularityCalculator.
                #
                # The old lookup arguments are intentionally passed as
                # None because they are no longer required by the
                # database analyzer.
                # ------------------------------------------------------

                result = self.compute_circularity_for_single_fu(
                    fu_dict=fu_dict,
                    mass_lookup=None,
                    energy_lookup=None,
                    water_lookup=None,
                    dry_mass_lookup=None,
                    ced_total_dict=None,
                    ced_renewable_dict=None,
                    ced_nonrenewable_dict=None,
                    isic_lookup_classified=isic_lookup_classified,
                    location_lookup_classified=location_lookup_classified,
                )

                if result is not None:
                    results.append(result)

                # ------------------------------------------------------
                # Interim saving
                #
                # Keep original behavior, but only save when a positive
                # save_interval is supplied.
                # ------------------------------------------------------

                if (
                    save_interval
                    and save_interval > 0
                    and len(results) % save_interval == 0
                ):
                    interim_df = pd.DataFrame(results)

                    output_folder = os.path.join(
                        "results",
                        "csv",
                        f"{self.project_name}",
                    )

                    os.makedirs(
                        output_folder,
                        exist_ok=True,
                    )

                    interim_filename = os.path.join(
                        output_folder,
                        f"circularity_interim_{len(results)}.csv",
                    )

                    interim_df.to_csv(
                        interim_filename,
                        index=False,
                    )

                    logger.info(
                        "Saved interim results to %s: %d processes",
                        interim_filename,
                        len(results),
                    )

            except Exception:
                logger.exception(
                    "Error processing activity index %d",
                    i + 1,
                )
                continue

        # --------------------------------------------------------------
        # Final statistics
        # --------------------------------------------------------------

        total_time = time.time() - start_time

        logger.info(
            "Sequential processing completed in %.1f minutes",
            total_time / 60.0,
        )

        if len(functional_units) > 0:
            success_rate = (
                len(results)
                / len(functional_units)
                * 100
            )

            logger.info(
                "Success rate: %d/%d (%.1f%%)",
                len(results),
                len(functional_units),
                success_rate,
            )

        else:
            logger.warning(
                "No functional units were processed."
            )

        return pd.DataFrame(results)