import os
import json
import time

from collections import defaultdict
from typing import Dict, Tuple, Any, List, Optional
import numbers
import logging

import ast

import numpy as np # type: ignore
import pandas as pd # type: ignore

import bw2data as bd # type: ignore
import bw2calc as bc # type: ignore

logger = logging.getLogger(__name__)


class CircularityCalculator:
    """
    Class to compute circularity indicators based on inventory results.
    """

    ALLOWED_MASS_STRATEGIES = {"full_mass", "water_mass", "dry_mass"}
    ALLOWED_ENERGY_STRATEGIES = {"full_energy", "renewable_energy", "non-renewable_energy"}

    def __init__(self, project_name, excluded_flows, valuable_water_compartments, exclude_water=True, technosphere_db_name=None, biosphere_db_name=None, mass_strategy=None, energy_strategy=None):
        self.project_name = project_name
        self.EXCLUDED_FLOWS = excluded_flows
        self.VALUABLE_WATER_COMPARTMENTS = valuable_water_compartments
        self.EXCLUDE_WATER = exclude_water
        self.technosphere_db_name = technosphere_db_name
        self.biosphere_db_name = biosphere_db_name
        # Validate and set strategies (provide sensible defaults)
        self.mass_strategy = mass_strategy or "full_mass"
        self.energy_strategy = energy_strategy or "full_energy"
        if self.mass_strategy not in self.ALLOWED_MASS_STRATEGIES:
            raise ValueError(f"Unknown mass_strategy: {self.mass_strategy}. Allowed: {self.ALLOWED_MASS_STRATEGIES}")
        if self.energy_strategy not in self.ALLOWED_ENERGY_STRATEGIES:
            raise ValueError(f"Unknown energy_strategy: {self.energy_strategy}. Allowed: {self.ALLOWED_ENERGY_STRATEGIES}")

    def set_strategies(self, mass_strategy: Optional[str] = None, energy_strategy: Optional[str] = None):
        """Runtime setter to override strategies (optional per-run control)."""
        if mass_strategy:
            if mass_strategy not in self.ALLOWED_MASS_STRATEGIES:
                raise ValueError(f"Unknown mass_strategy: {mass_strategy}")
            self.mass_strategy = mass_strategy
            logger.info("Mass strategy set to %s", mass_strategy)
        if energy_strategy:
            if energy_strategy not in self.ALLOWED_ENERGY_STRATEGIES:
                raise ValueError(f"Unknown energy_strategy: {energy_strategy}")
            self.energy_strategy = energy_strategy
            logger.info("Energy strategy set to %s", energy_strategy)

    def should_exclude_flow(self, unit, name, categories, flow_type):
        """
        Determine if a flow should be excluded from circularity calculations.
        """
        if isinstance(categories, str):
            try:
                categories = ast.literal_eval(categories)
            except:
                categories = ()

        # 1. Excluded emissions (dependant of the list in the main notebook e.g., BOD)
        is_excluded_emission = False
        if flow_type == 'emission' and unit == 'kilogram':
            for excluded_flow in self.EXCLUDED_FLOWS:
                if excluded_flow in name:
                    is_excluded_emission = True
                    break

        # Exclude water flows expressed in m³, in ei 3.11 cutoff, this concerns 23 elementary flows - when we add the cutoff activities, 2 additional wastewater flows may be concerned: ""
        is_water_flow = (
            (unit or '').lower() == 'cubic meter'
            and 'water' in name.lower()
            # and flow_type in ('natural resource', 'emission') # if we want to focus on initial biosphere flows
        )

        # Use self.exclude_water to decide whether to include the is_water_flow condition
        return (

            ('becquerel' in (unit or '').lower())
            or 'volume occupied' in name.lower()
            or (
                categories and len(categories) > 1
                and categories[0] == 'natural resource'
                and categories[1] == 'biotic' # we exclude wood but mass of these flows is obtained throughout kg_factors
                and unit in ('megajoule', 'kilowatt hour')
            )
            or is_excluded_emission
            or (self.EXCLUDE_WATER and is_water_flow)  # Only include is_water_flow if self.exclude_water is True
        )


    @staticmethod
    def _subcomp_from_categories(categories):
        """
        Extract subcompartment from categories tuple.
        """
        if not categories or not isinstance(categories, tuple):
            return "unknown"

        if len(categories) > 1:
            return categories[1]
        elif len(categories) == 1:
            return categories[0]
        else:
            return "unknown"

    def build_combined_lookup(self):
        """
        Build combined lookup for both mass and energy properties in one pass.
        """
        biosphere_db = bd.Database(self.biosphere_db_name)
        techno_to_bio_map = {}

        for flow in biosphere_db:
            categories = flow.get('categories', ())
            if categories and categories[0] == 'technosphere' and len(categories) > 1:
                location = categories[1]
                key = (flow.get('name', ''), location)
                techno_to_bio_map[key] = (flow.get('name', ''), categories)

        mass_lookup = {}
        energy_lookup = {}
        water_lookup = {}
        dry_mass_lookup = {}
        db = bd.Database(self.technosphere_db_name)

        for act in db: # ⚠️ we are recovering the exchange properties from the tech db !
            for exc in act.exchanges():
                properties = exc.get('properties', {})
                wet_mass = properties.get('wet mass', {}) # total mass
                dry_mass = properties.get('dry mass', {}) # dry mass (without water molecules)
                water_content = properties.get('water content', {}) # water content
                energy_content = properties.get('energy content', {})
                heating_value = properties.get('heating value, net', {})
                lhv_MJperkg = properties.get('lower heating value MJ per kg', {})
                gross_energy_per_dry_mass = properties.get('gross energy, feed per kg of dry mass', {})

                # mass
                has_wet_mass = wet_mass and wet_mass.get('amount') is not None
                has_water_content = water_content and water_content.get('amount') is not None
                has_dry_mass = dry_mass and dry_mass.get('amount') is not None
                # energy
                has_energy_content = energy_content and energy_content.get('amount') is not None
                has_heating_value = heating_value and heating_value.get('amount') is not None
                has_specific_lhv = lhv_MJperkg and lhv_MJperkg.get('amount') is not None # in MJ per kg
                has_GEperdrymass = gross_energy_per_dry_mass and gross_energy_per_dry_mass.get('amount') is not None

                if not (has_wet_mass or has_water_content or has_dry_mass or has_heating_value or has_energy_content or has_specific_lhv or has_GEperdrymass):
                    continue

                if exc.get('type') == 'biosphere':
                    biosphere_flow = exc.input
                    if biosphere_flow is None:
                        continue
                    categories = biosphere_flow.get('categories', ())
                    cat = CircularityCalculator._subcomp_from_categories(categories)
                    key = (biosphere_flow.get('name', ''), cat)
                else:
                    flow_name = exc['name']
                    location = act['location']
                    map_key = (flow_name, location)

                    if map_key in techno_to_bio_map:
                        bio_name, categories = techno_to_bio_map[map_key]
                        cat = CircularityCalculator._subcomp_from_categories(categories)
                        key = (bio_name, cat)
                    else:
                        key = (flow_name, "technosphere")
                        
                # --- store wet mass --- (total mass)
                if wet_mass and wet_mass.get('amount') is not None:
                    mass_lookup[key] = (wet_mass.get('amount'), wet_mass.get('unit'))
                # --- store water mass ---  
                if (
                    wet_mass and wet_mass.get('amount') is not None and
                    water_content and water_content.get('amount') is not None
                ):
                    water_mass_amount = wet_mass.get('amount') * water_content.get('amount')
                    water_lookup[key] = (water_mass_amount, wet_mass.get('unit'))
                # --- store dry mass ---   
                if dry_mass and dry_mass.get('amount') is not None:
                    dry_mass_lookup[key] = (dry_mass.get('amount'), dry_mass.get('unit'))
                # --- store energy content --- 
                if energy_content and energy_content.get('amount') is not None:
                    energy_lookup[key] = (energy_content.get('amount'), energy_content.get('unit'))
                elif heating_value and heating_value.get('amount') is not None:
                    energy_lookup[key] = (heating_value.get('amount'), heating_value.get('unit'))
                elif (
                    lhv_MJperkg and lhv_MJperkg.get('amount') is not None and
                    wet_mass and wet_mass.get('amount') is not None
                ):
                    energy_content_amount = wet_mass.get('amount') * lhv_MJperkg.get('amount')
                    energy_lookup[key] = (energy_content_amount, 'megajoule')

                elif (
                    gross_energy_per_dry_mass and gross_energy_per_dry_mass.get('amount') is not None and
                    dry_mass and dry_mass.get('amount') is not None
                ):
                    energy_content_amount = dry_mass.get('amount') * gross_energy_per_dry_mass.get('amount')
                    energy_lookup[key] = (energy_content_amount, 'megajoule')
                    
        return mass_lookup, water_lookup, dry_mass_lookup, energy_lookup

    def build_ced_energy_lookup(self):
        ced_total = defaultdict(float)
        ced_renewable = defaultdict(float)
        ced_nonrenewable = defaultdict(float)
        
        renewable_keywords = ["renewable", "wind", "solar", "water", "biomass"]
        nonrenewable_keywords = ["non-renewable", "fossil", "nuclear", "primary forest"]

        # First, collect all CED methods with their priority
        methods_with_priority = []
        
        for method in bd.methods:
            method_tuple = tuple(method)

            if not any("cumulative energy demand" in str(x).lower() for x in method_tuple):
                continue

            method_name = " ".join(method_tuple).lower()

            # Assign priority: total (1) > non-renewable (2) > renewable (3) > keyword-based (4)
            if "total" in method_name:
                priority = 1
                category = "total"
            elif "non-renewable" in method_name:
                priority = 2
                category = "nonrenewable"
            elif "renewable" in method_name:
                priority = 3
                category = "renewable"
            elif any(kw in method_name for kw in nonrenewable_keywords):
                priority = 4
                category = "nonrenewable"
            elif any(kw in method_name for kw in renewable_keywords):
                priority = 5
                category = "renewable"
            else:
                continue  # Skip uncategorized methods
                
            methods_with_priority.append((priority, category, method_tuple, method_name))

        # Sort by priority so higher priority methods are processed first
        methods_with_priority.sort(key=lambda x: x[0])
        
        # Track which flows have been assigned for each category
        assigned_total = set()
        assigned_nonrenewable = set()
        assigned_renewable = set()

        for priority, category, method_tuple, method_name in methods_with_priority:
            try:
                cf_data = bd.Method(method_tuple).load()
            except Exception:
                continue

            for flow_key, cf in cf_data:
                if not cf:
                    continue
                try:
                    flow = bd.get_activity(flow_key)
                except Exception:
                    continue

                flow_name = flow.get('name')
                categories = flow.get('categories', ())
                subc = CircularityCalculator._subcomp_from_categories(categories)
                lookup_key = (flow_name, subc)

                if category == "total":
                    # Only add if this flow hasn't been assigned to total yet
                    if lookup_key not in assigned_total:
                        ced_total[lookup_key] += cf
                        assigned_total.add(lookup_key)
                        
                elif category == "nonrenewable":
                    # Add to nonrenewable and total only if not already assigned
                    if lookup_key not in assigned_nonrenewable:
                        ced_nonrenewable[lookup_key] += cf
                        assigned_nonrenewable.add(lookup_key)
                    if lookup_key not in assigned_total:
                        ced_total[lookup_key] += cf
                        assigned_total.add(lookup_key)
                        
                elif category == "renewable":
                    # Add to renewable and total only if not already assigned
                    if lookup_key not in assigned_renewable:
                        ced_renewable[lookup_key] += cf
                        assigned_renewable.add(lookup_key)
                    if lookup_key not in assigned_total:
                        ced_total[lookup_key] += cf
                        assigned_total.add(lookup_key)

        return dict(ced_total), dict(ced_renewable), dict(ced_nonrenewable)

    
    def get_property_factors(
        self,
        flow,
        mass_lookup,
        water_lookup,
        dry_mass_lookup,
        energy_lookup,
        ced_total_dict,
        ced_renewable_dict,
        ced_nonrenewable_dict):

        """
        Get both kg and MJ factors for a flow in one function call.
        """
        kg_factor = None # wet mass = total mass
        kg_water_factor = None
        kg_dry_mass_factor = None # to compare

        MJ_factor = None
        
        unit = flow.get('unit', '')
        name = flow.get('name', '').lower()
        categories = flow.get('categories', ())
        f_type = flow.get('type', '')

        if self.should_exclude_flow(unit, name, categories, f_type):
            return None, None, None, None, None, None, None

        if unit == 'kilogram':
            kg_factor = 1.0
        elif unit == 'megajoule':
            MJ_factor = 1.0
        elif unit == 'kilowatt hour':
            MJ_factor = 3.6

        # old working block
        # # Add specific condition for wood flows
        # if (categories and len(categories) > 1 and
        #     categories[0] == 'natural resource' and
        #     categories[1] == 'biotic' and
        #     'wood' in name):
        #     if unit == 'cubic meter':
        #         MJ_factor = 10000  # MJ/m³ for wood according to average density in https://www.engineeringtoolbox.com/wood-combustion-heat-d_372.html
        #     elif unit == 'kilogram':
        #         MJ_factor = 20  # MJ/kg for wood according to https://www.engineeringtoolbox.com/wood-combustion-heat-d_372.html

        # new block for wood, must check ei property for density
        flow_props = flow.get('properties', {})
        if (categories and len(categories) > 1 and
            categories[0] == 'natural resource' and
            categories[1] == 'biotic' and
            'wood' in name):

            # Try real properties first (density x net heating value)
            density_props    = flow_props.get('density', {})
            density_amount   = density_props.get('amount') if density_props else None

            nhv_props        = flow_props.get('heating value, net', {})
            nhv_amount       = nhv_props.get('amount') if nhv_props else None

            lhv_per_kg_props = flow_props.get('lower heating value MJ per kg', {})
            lhv_per_kg       = lhv_per_kg_props.get('amount') if lhv_per_kg_props else None

            if unit == 'cubic meter':
                if density_amount is not None and nhv_amount is not None:
                    MJ_factor = density_amount * nhv_amount          # MJ/m3
                elif density_amount is not None and lhv_per_kg is not None:
                    MJ_factor = density_amount * lhv_per_kg          # MJ/m3
                else:
                    MJ_factor = 10960  # ecoinvent Tab. 2.3 average (hard 12740, soft 9180) MJ/m3 atro 
                    # ei_3.12 doc: https://www.researchgate.net/publication/263239305_Implementation_of_Life_Cycle_Impact_Assessment_Methods_ecoinvent_report_No_3_v22
                    # ei_3.12 example without the prop: https://ecoquery.ecoinvent.org/3.12/cutoff/dataset/2850/exchanges
            elif unit == 'kilogram':
                if nhv_amount is not None:
                    MJ_factor = nhv_amount                            # MJ/kg
                elif lhv_per_kg is not None:
                    MJ_factor = lhv_per_kg                            # MJ/kg
                else:
                    MJ_factor = 20.0   # ecoinvent Tab. 2.3 average (hard 19.61, soft 20.40) MJ/kg atro
        
        # --- Direct properties (from flow metadata) ---
        flow_props = flow.get('properties', {}) # If the flow is not in the technosphere database, new property digging is needed to get the mass and energy factors from the flow properties.
        
        if not kg_factor and 'wet mass' in flow_props:
            kg_factor = flow_props['wet mass'].get('amount')
            
        if not MJ_factor and 'heating value, net' in flow_props:
            MJ_factor = flow_props['heating value, net'].get('amount')

        # --- Lookup keys ---
        subc = CircularityCalculator._subcomp_from_categories(categories)
        flow_name = flow.get('name', '')

        lookup_key = (flow_name, subc)

        # --- CED lookups ---
        if self.energy_strategy == "non-renewable_energy":
            if MJ_factor is None and lookup_key in ced_nonrenewable_dict:
                MJ_factor = ced_nonrenewable_dict[lookup_key]
        elif self.energy_strategy == "full_energy":
            if MJ_factor is None and lookup_key in ced_total_dict:
                MJ_factor = ced_total_dict[lookup_key]
        elif self.energy_strategy == "renewable_energy":
            # unchanged legacy behaviour: renewable CED is only surfaced
            # through MJ_renewable_factor, not through MJ_factor
            pass
        else:
            raise ValueError(f"Unknown energy_strategy: {self.energy_strategy}")
        
        MJ_renewable_factor = ced_renewable_dict.get(lookup_key, 0.0)
        MJ_nonrenewable_factor = ced_nonrenewable_dict.get(lookup_key, 0.0)

        # --- Technosphere lookups ---
        key_formats = [
            (flow_name, subc),
            (flow_name, "technosphere"),
        ]

        if len(categories) > 1:
            key_formats.append((flow_name, categories[1]))

        for key in key_formats:
            if not kg_factor and key in mass_lookup:
                kg_factor = mass_lookup[key][0]
            if not kg_water_factor and key in water_lookup:
                kg_water_factor = water_lookup[key][0]
            if not kg_dry_mass_factor and key in dry_mass_lookup:
                kg_dry_mass_factor = dry_mass_lookup[key][0]
            if not MJ_factor and key in energy_lookup:
                MJ_factor = energy_lookup[key][0]
        
            if kg_factor is not None and MJ_factor is not None and kg_water_factor is not None:
                break
                
    # # --- NEW: CAS number fallback for biosphere flows --- FUTURE
    # if (kg_factor is None or MJ_factor is None) and f_type == 'biosphere':
    #     cas_number = flow.get('CAS number')
    #     if cas_number:
    #         cas_props = self._get_properties_from_cas(cas_number) # API call...
    #         if not kg_factor and 'wet mass' in cas_props:
    #             kg_factor = cas_props['wet mass'].get('amount')
    #         if not MJ_factor and 'heating value' in cas_props:
    #             MJ_factor = cas_props['heating value'].get('amount')
    #         if not MJ_factor and 'heating value, net' in cas_props:
    #             MJ_factor = cas_props['heating value, net'].get('amount')
    #         # Add other properties as needed (e.g., dry_mass, energy_content, etc.)
    #         if not kg_water_factor and 'water content' in cas_props and kg_factor is not None:
    #             water_content = cas_props['water content'].get('amount')
    #             if water_content is not None:
    #                 kg_water_factor = kg_factor * water_content
    #         if not kg_dry_mass_factor and 'dry mass' in cas_props:
    #             kg_dry_mass_factor = cas_props['dry mass'].get('amount')

        #--- The water content is badly reported, there is some case where 'kg_dry_mass' is not equal to 'kg_dry_mass_factor_2' ---
        # --- Derived dry mass (wet - water) ---
        kg_dry_mass_factor_2 = None

        if kg_factor is not None:
            if kg_water_factor is not None:
                kg_dry_mass_factor_2 = kg_factor - kg_water_factor # is 1000 for some elementary flows => water content not reported
            else:
                kg_dry_mass_factor_2 = kg_factor

        return kg_factor, kg_water_factor, kg_dry_mass_factor, kg_dry_mass_factor_2, MJ_factor, MJ_renewable_factor, MJ_nonrenewable_factor
    
    def select_mass_factor(self,
                        kg_factor,
                        kg_water_factor,
                        kg_dry_mass_factor,
                        kg_dry_mass_factor_2):
        """
        Select mass factor according to chosen strategy.
        """

        if self.mass_strategy == "full_mass":
            return kg_factor

        elif self.mass_strategy == "water_mass":
            return kg_water_factor

        elif self.mass_strategy == "dry_mass":
            # Prefer explicit dry mass property
            if kg_dry_mass_factor is not None:
                return kg_dry_mass_factor
            else:
                return kg_dry_mass_factor_2

        else:
            raise ValueError(f"Unknown mass strategy: {self.mass_strategy}")
        
    def select_energy_factor(self, MJ_factor, MJ_renewable_factor, MJ_nonrenewable_factor):
        if self.energy_strategy == "full_energy":
            return MJ_factor
        elif self.energy_strategy == "renewable_energy":
            return MJ_renewable_factor
        elif self.energy_strategy == "non-renewable_energy":
            return MJ_factor
        else:
            raise ValueError(f"Unknown energy_strategy: {self.energy_strategy}")

    def compare_dry_mass_factors(self, detailed_flows_df, tolerance=1e-6):
        """
        Compare explicit dry mass vs computed dry mass (wet - water).
        """

        comparison_results = []

        for _, row in detailed_flows_df.iterrows():

            explicit = row.get("kg_dry_mass_factor")
            computed = row.get("kg_dry_mass_factor_2")

            if explicit is None or computed is None:
                continue

            difference = explicit - computed
            relative_error = difference / explicit if explicit != 0 else None

            comparison_results.append({
                "Flow Name": row["Flow Name"],
                "Explicit Dry Mass": explicit,
                "Computed Dry Mass": computed,
                "Absolute Difference": difference,
                "Relative Error": relative_error
            })

        return pd.DataFrame(comparison_results)

    @staticmethod
    def get_inventory_flows(setup_name=None, functional_unit=None):
        """
        Retrieve inventory flows from either a calculation setup or a functional unit.
        
        Parameters:
        -----------
        setup_name : str, optional
            Name of the calculation setup in Brightway2
        functional_unit : dict, optional
            Direct functional unit like {process.key: 1}
        
        Returns:
        --------
        pd.DataFrame: Inventory flows
        """
        all_flows = []
        
        # Get functional units from either setup or direct input
        if setup_name:
            setup = bd.calculation_setups[setup_name]
            functional_units = setup['inv']
        elif functional_unit:
            functional_units = [functional_unit]
        else:
            raise ValueError("Either setup_name or functional_unit must be provided")
        
        for fu_dict in functional_units:
            lca = bc.LCA(fu_dict)
            lca.lci()
            
            inventory = lca.inventory.sum(axis=1)
            if hasattr(inventory, 'A1'):
                inventory = inventory.A1
            
            # Flatten to 1D if needed
            if hasattr(inventory, 'flatten'):
                inventory = inventory.flatten()
            elif hasattr(inventory, 'toarray'):
                inventory = inventory.toarray().flatten()
            
            reverse_dict = {v: k for k, v in lca.biosphere_dict.items()}
            non_zero_indices = np.where(np.abs(inventory) > 1e-20)[0]
            
            for idx in non_zero_indices:
                try:
                    amount = float(inventory[idx])
                except (TypeError, ValueError):
                    # Handle case where inventory[idx] is an array
                    if hasattr(inventory[idx], 'item'):
                        amount = float(inventory[idx].item())
                    else:
                        amount = 0.0
                
                flow_key = reverse_dict.get(idx)
                if flow_key:
                    try:
                        flow = bd.get_activity(flow_key)
                        all_flows.append({
                            'Flow Key': flow_key,
                            'Flow Name': flow.get('name', 'Unknown'),
                            'Amount': amount,
                            'Unit': flow.get('unit', 'Unknown'),
                            'Categories': flow.get('categories', ()),
                            'Type': flow.get('type', 'Unknown')
                        })
                    except Exception as e:
                        logger.exception("Error getting flow %s: %s", flow_key, e)
        
        return pd.DataFrame(all_flows) if all_flows else pd.DataFrame()

    def compute_circularity_efficiency_variables(
        self,
        flows_df=None,
        setup_name=None,
        functional_unit=None,
        save_csv=True,
        mass_strategy: Optional[str]=None, 
        energy_strategy: Optional[str]=None):
        """
        Compute circularity indicators.
        
        Can accept either:
        - flows_df: A pre-computed DataFrame of flows
        - setup_name: Name of a Brightway2 calculation setup
        - functional_unit: Direct functional unit like {process.key: 1}
        
        If flows_df is not provided, it will be computed from setup_name or functional_unit.
        """
        orig_mass, orig_energy = self.mass_strategy, self.energy_strategy
        
        try:
            if mass_strategy or energy_strategy:
                self.set_strategies(mass_strategy=mass_strategy, energy_strategy=energy_strategy)
            
            # Get flows if not provided
            if flows_df is None:
                if setup_name:
                    flows_df = self.get_inventory_flows(setup_name=setup_name)
                elif functional_unit:
                    flows_df = self.get_inventory_flows(functional_unit=functional_unit)
                else:
                    raise ValueError("Must provide either flows_df, setup_name, or functional_unit")
            
            if flows_df.empty:
                logger.warning("No flows found in inventory")
                return None, None, None, None
            
            # Build lookups if not already done
            if not hasattr(self, "_combined_lookup"):
                self._combined_lookup = self.build_combined_lookup()
            
            mass_lookup, water_lookup, dry_mass_lookup, energy_lookup = self._combined_lookup
            
            if not hasattr(self, "_ced_lookup"):
                self._ced_lookup = self.build_ced_energy_lookup()
            
            ced_total_dict, ced_renewable_dict, ced_nonrenewable_dict = self._ced_lookup
            
            # Initialize dictionaries
            V = defaultdict(float)
            Ri = defaultdict(float)
            Rr = defaultdict(float)
            Er = defaultdict(float)
            W = defaultdict(float)
            
            V_mass = Ri_mass = Rr_mass = Er_mass = W_mass = 0.0
            V_energy = Ri_energy = Rr_energy = Er_energy = W_energy = 0.0
            
            detailed_flows = []
            flow_cache = {}
            
            # Cache flow lookups
            for flow_key in flows_df['Flow Key'].unique():
                if flow_key not in flow_cache:
                    try:
                        flow_cache[flow_key] = bd.get_activity(flow_key)
                    except Exception:
                        flow_cache[flow_key] = None
            
            # Add functional unit to Rr if setup_name provided
            if setup_name:
                try:
                    setup = bd.calculation_setups[setup_name]
                    functional_units = setup['inv']
                    
                    for fu_dict in functional_units:
                        for fu_key, fu_amount in fu_dict.items():
                            try:
                                fu_activity = bd.get_activity(fu_key)
                                fu_name = fu_activity.get('name', 'Unknown')
                                fu_unit = fu_activity.get('unit', 'Unknown')
                                fu_amount = abs(fu_amount)
                                
                                logger.info("Adding functional unit to Rr: %s %s of %s", fu_amount, fu_unit, fu_name)
                                
                                kg_factor, kg_water_factor, kg_dry_mass_factor, kg_dry_mass_factor_2, MJ_factor, MJ_renewable_factor, MJ_nonrenewable_factor = self.get_property_factors(
                                    fu_activity, mass_lookup, water_lookup, dry_mass_lookup, energy_lookup,
                                    ced_total_dict, ced_renewable_dict, ced_nonrenewable_dict
                                )
                                
                                selected_mass_factor = self.select_mass_factor(
                                    kg_factor, kg_water_factor, kg_dry_mass_factor, kg_dry_mass_factor_2
                                )
                                
                                selected_energy_factor = self.select_energy_factor(
                                    MJ_factor, MJ_renewable_factor, MJ_nonrenewable_factor
                                )
                                
                                Rr[fu_unit] += fu_amount
                                if selected_mass_factor:
                                    Rr_mass += fu_amount * selected_mass_factor
                                if selected_energy_factor:
                                    Rr_energy += fu_amount * selected_energy_factor
                                
                                detailed_flows.append({
                                    'Flow Key': fu_key,
                                    'Flow Name': f"FUNCTIONAL UNIT: {fu_name}",
                                    'Amount': fu_amount,
                                    'Unit': fu_unit,
                                    'Categories': "functional_unit",
                                    'Type': "technosphere",
                                    'Category': 'Recycled Outputs (Rr) - Functional Unit',
                                    'kg_Factor': kg_factor,
                                    'kg_water_factor': kg_water_factor,
                                    'kg_dry_mass_factor': kg_dry_mass_factor,
                                    'kg_dry_mass_factor_2': kg_dry_mass_factor_2,
                                    'Mass_Equivalent_kg': fu_amount * kg_factor if kg_factor else None,
                                    'Mass_Water_Equivalent_kg': fu_amount * kg_water_factor if kg_water_factor else None,
                                    'Mass_dry1_Equivalent_kg': fu_amount * kg_dry_mass_factor if kg_dry_mass_factor else None,
                                    'Mass_dry2_Equivalent_kg': fu_amount * kg_dry_mass_factor_2 if kg_dry_mass_factor_2 else None,
                                    'MJ_Factor': MJ_factor,
                                    'MJ_renewable_Factor': MJ_renewable_factor,
                                    'MJ_nonrenewable_Factor': MJ_nonrenewable_factor,
                                    'Energy_Equivalent_MJ': fu_amount * MJ_factor if MJ_factor else None,
                                    'Energy_renewable_MJ': fu_amount * MJ_renewable_factor if MJ_renewable_factor else None,
                                    'Energy_nonrenewable_MJ': fu_amount * MJ_nonrenewable_factor if MJ_nonrenewable_factor else None
                                })
                                
                            except Exception as e:
                                logger.exception("Error processing functional unit %s: %s", fu_key, e)
                except Exception as e:
                    logger.exception("Error getting functional unit information: %s", e)
            # Add functional unit to Rr if functional_unit provided directly
            if functional_unit and not setup_name:
                try:
                    # functional_unit is already a dict, not a list of dicts
                    for fu_key, fu_amount in functional_unit.items():
                        try:
                            fu_activity = bd.get_activity(fu_key)
                            fu_name = fu_activity.get('name', 'Unknown')
                            fu_unit = fu_activity.get('unit', 'Unknown')
                            fu_amount = abs(fu_amount)
                            
                            logger.info("Adding functional unit to Rr (direct FU): %s %s of %s", fu_amount, fu_unit, fu_name)
                            
                            kg_factor, kg_water_factor, kg_dry_mass_factor, kg_dry_mass_factor_2, MJ_factor, MJ_renewable_factor, MJ_nonrenewable_factor = self.get_property_factors(
                                fu_activity, mass_lookup, water_lookup, dry_mass_lookup, energy_lookup,
                                ced_total_dict, ced_renewable_dict, ced_nonrenewable_dict
                            )
                            
                            selected_mass_factor = self.select_mass_factor(
                                kg_factor, kg_water_factor, kg_dry_mass_factor, kg_dry_mass_factor_2
                            )
                            
                            selected_energy_factor = self.select_energy_factor(
                                MJ_factor, MJ_renewable_factor, MJ_nonrenewable_factor
                            )
                            
                            Rr[fu_unit] += fu_amount
                            if selected_mass_factor:
                                Rr_mass += fu_amount * selected_mass_factor
                            if selected_energy_factor:
                                Rr_energy += fu_amount * selected_energy_factor
                            
                            detailed_flows.append({
                                'Flow Key': fu_key,
                                'Flow Name': f"FUNCTIONAL UNIT: {fu_name}",
                                'Amount': fu_amount,
                                'Unit': fu_unit,
                                'Categories': "functional_unit",
                                'Type': "technosphere",
                                'Category': 'Recycled Outputs (Rr) - Functional Unit',
                                'kg_Factor': kg_factor,
                                'kg_water_factor': kg_water_factor,
                                'kg_dry_mass_factor': kg_dry_mass_factor,
                                'kg_dry_mass_factor_2': kg_dry_mass_factor_2,
                                'Mass_Equivalent_kg': fu_amount * kg_factor if kg_factor else None,
                                'Mass_Water_Equivalent_kg': fu_amount * kg_water_factor if kg_water_factor else None,
                                'Mass_dry1_Equivalent_kg': fu_amount * kg_dry_mass_factor if kg_dry_mass_factor else None,
                                'Mass_dry2_Equivalent_kg': fu_amount * kg_dry_mass_factor_2 if kg_dry_mass_factor_2 else None,
                                'MJ_Factor': MJ_factor,
                                'MJ_renewable_Factor': MJ_renewable_factor,
                                'MJ_nonrenewable_Factor': MJ_nonrenewable_factor,
                                'Energy_Equivalent_MJ': fu_amount * MJ_factor if MJ_factor else None,
                                'Energy_renewable_MJ': fu_amount * MJ_renewable_factor if MJ_renewable_factor else None,
                                'Energy_nonrenewable_MJ': fu_amount * MJ_nonrenewable_factor if MJ_nonrenewable_factor else None
                            })
                            
                        except Exception as e:
                            logger.exception("Error processing direct functional unit %s: %s", fu_key, e)
                except Exception as e:
                    logger.exception("Error processing direct functional unit: %s", e)
                    
            # Process flows
            for idx, row in flows_df.iterrows():
                try:
                    amount = float(row['Amount']) # type: ignore
                except (TypeError, ValueError):
                    if hasattr(row['Amount'], 'item'):
                        amount = float(row['Amount'].item())
                    else:
                        amount = 0.0
                
                unit = row['Unit']
                categories = row['Categories']
                flow_type = (row['Type'] or '').lower()
                flow_key = row['Flow Key']
                name = row['Flow Name']
                flow = flow_cache[flow_key]
                
                if flow is None:
                    continue
                
                if self.should_exclude_flow(unit, name, categories, flow_type):
                    continue
                
                kg_factor, kg_water_factor, kg_dry_mass_factor, kg_dry_mass_factor_2, MJ_factor, MJ_renewable_factor, MJ_nonrenewable_factor = self.get_property_factors(
                    flow, mass_lookup, water_lookup, dry_mass_lookup, energy_lookup,
                    ced_total_dict, ced_renewable_dict, ced_nonrenewable_dict
                )
                
                selected_mass_factor = self.select_mass_factor(
                    kg_factor, kg_water_factor, kg_dry_mass_factor, kg_dry_mass_factor_2
                )
                
                selected_energy_factor = self.select_energy_factor(
                    MJ_factor, MJ_renewable_factor, MJ_nonrenewable_factor
                )
                
                category = None
                mass_equiv = (amount) * kg_factor if kg_factor else None
                mass_water_equiv = (amount) * kg_water_factor if kg_water_factor else None
                mass_dry1_equiv = (amount) * kg_dry_mass_factor if kg_dry_mass_factor else None
                mass_dry2_equiv = (amount) * kg_dry_mass_factor_2 if kg_dry_mass_factor_2 else None
                energy_equiv = (amount) * MJ_factor if MJ_factor else None
                energy_renewable_equiv = (amount) * MJ_renewable_factor if MJ_renewable_factor else None
                energy_nonrenewable_equiv = (amount) * MJ_nonrenewable_factor if MJ_nonrenewable_factor else None
                
                # Avoid truth-checking 'categories' directly (can be a pandas Series)
                if 'resource' in flow_type or (
                    categories is not None and len(categories) and
                    any(isinstance(cat, str) and 'resource' in cat.lower() for cat in categories)
                ):
                    V[unit] += amount
                    if selected_mass_factor:
                        V_mass += amount * selected_mass_factor
                    if selected_energy_factor:
                        V_energy += amount * selected_energy_factor
                    category = 'Natural Resources (V)'
                    
                elif 'technosphere' in flow_type:
                    if amount > 0:
                        Ri[unit] += amount
                        if selected_mass_factor:
                            Ri_mass += amount * selected_mass_factor
                        if selected_energy_factor:
                            Ri_energy += amount * selected_energy_factor
                        category = 'Technosphere Inputs (Ri)'
                    else:
                        Rr[unit] += abs(amount)
                        if selected_mass_factor:
                            Rr_mass += abs(amount) * selected_mass_factor
                        if selected_energy_factor:
                            Rr_energy += abs(amount) * selected_energy_factor
                        category = 'Recycled Outputs (Rr)'

                # Cleaned emissions (Er) 
                elif 'emission' in flow_type:
                    if (name == 'Water' and isinstance(categories, (list, tuple)) and len(categories) >= 2
                        and isinstance(categories[0], str) and categories[0].lower() == 'water'
                        and any(isinstance(cat, str) and cat.lower().startswith(tuple(self.VALUABLE_WATER_COMPARTMENTS)) for cat in categories)): # type: ignore
                        
                        Er[unit] += (amount)
                        if selected_mass_factor:
                            Er_mass += (amount) * selected_mass_factor
                        if selected_energy_factor:
                            Er_energy += (amount) * selected_energy_factor
                        category = 'Cleaned Emissions (Er)'
                        
                    elif (unit == "kilogram" and flow.get("name") in self.EXCLUDED_FLOWS): # type: ignore
                        category = 'Excluded Emission'
                        continue
                    else:
                        W[unit] += (amount)
                        if selected_mass_factor:
                            W_mass += (amount) * selected_mass_factor
                        if selected_energy_factor:
                            W_energy += (amount) * selected_energy_factor
                        category = 'Waste Emissions (W)'
                
                detailed_flows.append({
                    'Flow Key': flow_key,
                    'Flow Name': name,
                    'Amount': amount,
                    'Unit': unit,
                    'Categories': str(categories),
                    'Type': flow_type,
                    'Category': category,
                    'Mass_Equivalent_kg': mass_equiv,
                    'Energy_Equivalent_MJ': energy_equiv,
                    'Mass_Water_Equivalent_kg': mass_water_equiv,
                    'Mass_dry1_Equivalent_kg': mass_dry1_equiv,
                    'Mass_dry2_Equivalent_kg': mass_dry2_equiv,
                    'Energy_renewable_MJ': energy_renewable_equiv,
                    'Energy_nonrenewable_MJ': energy_nonrenewable_equiv,
                    'kg_Factor': kg_factor,
                    'MJ_Factor': MJ_factor,
                    'kg_water_factor': kg_water_factor,
                    'kg_dry_mass_factor': kg_dry_mass_factor,
                    'kg_dry_mass_factor_2': kg_dry_mass_factor_2,
                    'MJ_renewable_Factor': MJ_renewable_factor,
                    'MJ_nonrenewable_Factor': MJ_nonrenewable_factor
                })
            
            # Create results DataFrame
            detailed_flows_df = pd.DataFrame(detailed_flows)
            
            if save_csv and not detailed_flows_df.empty:
                output_folder = os.path.join("results", "csv", f"{self.project_name}")
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                csv_filename = os.path.join(output_folder, "circularity_inventory.csv")
                detailed_flows_df.to_csv(csv_filename, index=False)
                logger.info("Detailed inventory saved to: %s", os.path.abspath(csv_filename))
            
            # Create summary DataFrame
            V_df = pd.DataFrame.from_dict(V, orient='index', columns=['Natural Resources'])
            Ri_df = pd.DataFrame.from_dict(Ri, orient='index', columns=['Technosphere Inputs'])
            Rr_df = pd.DataFrame.from_dict(Rr, orient='index', columns=['Recycled Outputs'])
            Er_df = pd.DataFrame.from_dict(Er, orient='index', columns=['Cleaned Emissions'])
            W_df = pd.DataFrame.from_dict(W, orient='index', columns=['Waste Emissions'])
            
            combined_df = pd.concat([V_df, Ri_df, Rr_df, Er_df, W_df], axis=1)
            combined_df = combined_df.infer_objects(copy=False).fillna(0) # type: ignore
            combined_df['Total Input (V+Ri)'] = combined_df['Natural Resources'] + combined_df['Technosphere Inputs']
            combined_df['Total functional (Er+Rr)'] = combined_df['Recycled Outputs'] + combined_df['Cleaned Emissions']
            combined_df['inefficiency (η-)'] = combined_df['Waste Emissions'] / combined_df['Total Input (V+Ri)']
            combined_df['efficiency (η+)'] = combined_df['Total functional (Er+Rr)'] / combined_df['Total Input (V+Ri)']
            
            # Add mass and energy rows
            if V_mass or Ri_mass or Rr_mass or Er_mass or W_mass:
                combined_df.loc['kilogram', ['Natural Resources', 'Technosphere Inputs', 'Recycled Outputs', 'Cleaned Emissions', 'Waste Emissions']] = [
                    V_mass, Ri_mass, Rr_mass, Er_mass, W_mass
                ]
                combined_df.loc['kilogram', 'Total Input (V+Ri)'] = V_mass + Ri_mass
                combined_df.loc['kilogram', 'Total functional (Er+Rr)'] = Er_mass + Rr_mass
                combined_df.loc['kilogram', 'inefficiency (η-)'] = (W_mass / (V_mass + Ri_mass)) if (V_mass + Ri_mass) else 0
                combined_df.loc['kilogram', 'efficiency (η+)'] = ((Er_mass + Rr_mass) / (V_mass + Ri_mass)) if (V_mass + Ri_mass) else 0
            
            if V_energy or Ri_energy or Rr_energy or Er_energy or W_energy:
                combined_df.loc['megajoule', ['Natural Resources', 'Technosphere Inputs', 'Recycled Outputs', 'Cleaned Emissions', 'Waste Emissions']] = [
                    V_energy, Ri_energy, Rr_energy, Er_energy, W_energy
                ]
                combined_df.loc['megajoule', 'Total Input (V+Ri)'] = V_energy + Ri_energy
                combined_df.loc['megajoule', 'Total functional (Er+Rr)'] = Er_energy + Rr_energy
                combined_df.loc['megajoule', 'inefficiency (η-)'] = (W_energy / (V_energy + Ri_energy)) if (V_energy + Ri_energy) else 0
                combined_df.loc['megajoule', 'efficiency (η+)'] = ((Er_energy + Rr_energy) / (V_energy + Ri_energy)) if (V_energy + Ri_energy) else 0
            
            return combined_df, combined_df['inefficiency (η-)'], combined_df['efficiency (η+)'], detailed_flows_df
            
        finally:
            self.mass_strategy, self.energy_strategy = orig_mass, orig_energy

    def compute_circularity_EMF_indicators(self, flows_df, setup_name=None, functional_unit=None): # based on the Ellen Mac Arthur Method, simple version available in https://www.ellenmacarthurfoundation.org/material-circularity-indicator
        """
        Compute modified circularity indicators with formulas:
        LFI = (W + V) / (2*(Ri + V))
        CFI = (Rr + Er + Ri) / (2*(Ri + V))
        """
        if flows_df.empty:
            return None, None, None

        # Get the standard circularity results first
        results_df, LFI, CFI, detailed_flows_df = self.compute_circularity_efficiency_variables(flows_df, save_csv=False, setup_name=setup_name, functional_unit=functional_unit)
        
        if results_df is None:
            return None, None, None

        # Calculate modified indicators for each unit
        modified_results = {}
        
        for unit in results_df.index:
            V = results_df.loc[unit, 'Natural Resources']
            Ri = results_df.loc[unit, 'Technosphere Inputs']
            Rr = results_df.loc[unit, 'Recycled Outputs']
            Er = results_df.loc[unit, 'Cleaned Emissions']
            W = results_df.loc[unit, 'Waste Emissions']

            # Calculate modified LFI and CFI
            denominator = 2 * (Ri + V)
            
            if denominator != 0: # type: ignore
                modified_LFI = (W + V) / denominator
                modified_CFI = (Rr + Er + Ri) / denominator
            else:
                modified_LFI = 0
                modified_CFI = 0

            modified_results[unit] = {
                'simplified_LFI': modified_LFI, # linear / total use
                'CFI': modified_CFI, # circular / total use
                'Natural_Resources_V': V,
                'Technosphere_Inputs_Ri': Ri,
                'Recycled_Outputs_Rr': Rr,
                'Cleaned_Emissions_Er': Er,
                'Waste_Emissions_W': W
            }

        # Create modified results DataFrame
        modified_df = pd.DataFrame(modified_results).T
        
        return modified_df, results_df, detailed_flows_df


    def compute_iso59020_recycled_input_rate(self, flows_df, setup_name=None, functional_unit=None):
        """
        Compute recycled input rate (%) based on ISO 59020 logic.
    
        ISO 59020 (Circular economy — Measuring and assessing circularity performance)
        recommends indicators based on the share of secondary material
        in total material input.
    
        Recycled Input Rate (%) = Ri / (Ri + V) * 100
    
        where:
            Ri = Recycled inputs
            V  = Virgin (natural resource) inputs
        """
    
        if flows_df.empty:
            return None, None, None
    
        results_df, LFI, CFI, detailed_flows_df = \
            self.compute_circularity_efficiency_variables(
                flows_df,
                save_csv=False,
                setup_name=setup_name, 
                functional_unit=functional_unit
            )
    
        if results_df is None:
            return None, None, None
    
        iso_results = {}
    
        for unit in results_df.index:
    
            V = results_df.loc[unit, 'Natural Resources']
            Ri = results_df.loc[unit, 'Technosphere Inputs']
    
            total_input = V + Ri
    
            if total_input != 0: # type: ignore
                recycled_input_rate = (Ri / total_input) * 100
            else:
                recycled_input_rate = 0
    
            iso_results[unit] = {
                "Recycled_Input_Rate_percent": recycled_input_rate,
                "Virgin_Input_V": V,
                "Recycled_Input_Ri": Ri,
                "Total_Material_Input": total_input
            }
    
        iso_df = pd.DataFrame(iso_results).T
    
        return iso_df, results_df, detailed_flows_df

    def compute_iso59020_recycled_output_rate(self, flows_df, setup_name=None, functional_unit=None):
        """
        Compute recycled output rate (%) based on ISO 59020 logic.
    
        ISO 59020 (Circular economy — Measuring and assessing circularity performance)
        recommends indicators based on the share of materials recovered
        for circular use in total material outflows.
    
        Recycled Output Rate (%) = Rr / (Rr + W) * 100
    
        where:
            Rr = Recycled outputs
            W  = Waste emissions
    
        Note:
            Cleaned emissions (Er) are ignored in this approach,
            as they do not represent material reintegration
            into the technosphere.
        """
    
        if flows_df.empty:
            return None, None, None
    
        results_df, LFI, CFI, detailed_flows_df = \
            self.compute_circularity_efficiency_variables(
                flows_df,
                save_csv=False,
                setup_name=setup_name, 
                functional_unit=functional_unit
            )
    
        if results_df is None:
            return None, None, None
    
        iso_results = {}
    
        for unit in results_df.index:
    
            Rr = results_df.loc[unit, 'Recycled Outputs']
            W = results_df.loc[unit, 'Waste Emissions']
    
            total_output = Rr + W
    
            if total_output != 0: # type: ignore
                recycled_output_rate = (Rr / total_output) * 100
            else:
                recycled_output_rate = 0
    
            iso_results[unit] = {
                "Recycled_Output_Rate_percent": recycled_output_rate,
                "Recycled_Output_Rr": Rr,
                "Waste_Output_W": W,
                "Total_Material_Output": total_output
            }
    
        iso_df = pd.DataFrame(iso_results).T
    
        return iso_df, results_df, detailed_flows_df
    #--- Example ---# A.3.5 indicator definition, but can be adapted for other indicators that require detailed flow analysis with specific mass strategies and component categorization.
    # taken from https://www.cencenelec.eu/media/CEN-CENELEC/Events/Events/2024/2024-07-10_CE_Event_NEN/final_p2_hkroder-iso-59020.pdf
    def compute_A_3_5_perc_actual_recirc_biological(self, flows_df, setup_name=None, functional_unit=None):
        """
        Compute indicator A.3.5 (mass): "Per cent actual recirculation of outflow in the biological cycle".

        Formula (mass, dry-mass filter): Er / (W + Rr + Er)

        Behaviour:
        - Forces mass_strategy='dry_mass' (ignores water content).
        - Uses compute_circularity_efficiency_variables(...) with a per-call override.
        - Returns a dict: { 'indicator': float (0-1), 'Er_kg': float, 'Rr_kg': float, 'W_kg': float, 'denominator': float }
        """
        try:
            # Get circularity results using dry mass (ignore water)
            combined_df, ineff, eff, detailed_flows_df = self.compute_circularity_efficiency_variables(
                flows_df,
                save_csv=False,
                setup_name=setup_name,
                functional_unit=functional_unit,
                mass_strategy="dry_mass" # full mass may be used if the water content is included
            )

            if detailed_flows_df is None or detailed_flows_df.empty:
                logger.info("A.3.5: detailed flows empty, returning zero indicator")
                return {"indicator": 0.0, "Er_kg": 0.0, "Rr_kg": 0.0, "W_kg": 0.0, "denominator": 0.0}

            # Ensure numeric column and absolute values (we treat outflows as positives in mass equivalents)
            if "Mass_Equivalent_kg" not in detailed_flows_df.columns:
                logger.warning("A.3.5: detailed flows missing 'Mass_Equivalent_kg' column")
                return {"indicator": 0.0, "Er_kg": 0.0, "Rr_kg": 0.0, "W_kg": 0.0, "denominator": 0.0}

            mass_series = detailed_flows_df["Mass_Equivalent_kg"].fillna(0).abs()

            # Define category masks (covers functional-unit Rr entries which include 'Recycled Outputs')
            cat_series = detailed_flows_df.get("Category", "").astype(str) # type: ignore

            er_mask = cat_series.str.contains("Cleaned Emissions", na=False)
            rr_mask = cat_series.str.contains("Recycled Outputs", na=False)
            w_mask = cat_series.str.contains("Waste Emissions", na=False)

            Er_kg = float(mass_series[er_mask].sum())
            Rr_kg = float(mass_series[rr_mask].sum())
            W_kg = float(mass_series[w_mask].sum())

            denominator = Er_kg + Rr_kg + W_kg
            indicator = (Er_kg / denominator) if denominator > 0 else 0.0

            logger.info("A.3.5 (dry_mass) computed: indicator=%.4f (Er=%.4f, Rr=%.4f, W=%.4f)", indicator, Er_kg, Rr_kg, W_kg)

            return {
                "indicator": indicator,
                "Er_kg": Er_kg,
                "Rr_kg": Rr_kg,
                "W_kg": W_kg,
                "denominator": denominator
            }

        except Exception as e:
            logger.exception("Error computing A.3.5 indicator: %s", e)
            return {"indicator": 0.0, "Er_kg": 0.0, "Rr_kg": 0.0, "W_kg": 0.0, "denominator": 0.0}