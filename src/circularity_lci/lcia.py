"""
Life Cycle Impact Assessment (LCIA) module for circularity_lci.

Creates Brightway LCIA methods that EXACTLY mirror the flow tracking logic
from circularity_calculator.py's compute_circularity_efficiency_variables function.

Method Types Created:
- V: Cumulative Natural Resources Inputs
- ❌ V+Ri: Cumulative Resources Inputs (natural + technosphere inputs)
- V+Ri+Rr+Er: Net Cumulative Resources Inputs (includes recycled outputs)
- W: Cumulative Waste Outputs
- W+Er: Total Emissions (includes cleaned emissions)
"""
import ast

import bw2data as bd  # type: ignore
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Set
import logging

from .circularity_calculator import CircularityCalculator

logger = logging.getLogger(__name__)


class LCIAMethodBuilder(CircularityCalculator):
    """
    Build LCIA methods for circularity assessment by EXACTLY mirroring the logic
    from CircularityCalculator.compute_circularity_efficiency_variables().

    Flow categorization (mirrors calculator exactly):
    - V: Natural resource flows (type contains 'resource' OR categories contain 'resource')
    - Ri: Technosphere flows treated as inputs (positive CF)
    - Rr: Technosphere flows treated as recycled outputs (negative CF)
    - Er: Valuable water emissions (cleaned emissions, negative CF)
    - W: Waste emissions (other emissions, positive CF)
    """

    # Mass strategies (including the computed dry_mass_2)
    MASS_STRATEGIES = ["full_mass", "water_mass", "dry_mass", "dry_mass_2"]
    
    # Energy strategies
    ENERGY_STRATEGIES = ["full_energy", "renewable_energy", "non_renewable_energy"]

    # Method type definitions with their flow categories and CF signs
    METHOD_TYPES = {
        "V": {
            "name": "Cumulative Natural Resources Inputs",
            "categories": ["V"],
            "cf_signs": {"V": 1},
            "description": "Sum of all natural resource inputs"
        },
        "V+R+Er": {
            "name": "Net Cumulative Resources Inputs",
            "categories": ["V", "technosphere", "Er"],
            "cf_signs": {"V": 1, "technosphere": 1, "Er": -1},
            "description": "Natural resources + all technosphere flows (Ri positive, Rr negative via LCI) + All usefull elementary flow outputs"
        },
        "W": {
            "name": "Cumulative Waste Outputs",
            "categories": ["W"],
            "cf_signs": {"W": 1},
            "description": "Sum of all waste emissions"
        },
        "W+Er": {
            "name": "Total Emissions",
            "categories": ["W", "Er"],
            "cf_signs": {"W": 1, "Er": 1},
            "description": "Sum of all emissions"
        }
    }

    def __init__(
        self,
        project_name: str,
        excluded_flows: List[str],
        valuable_water_compartments: List[str],
        exclude_water: bool = True,
        technosphere_db_name: str = "ecoinvent-3.11-cutoff",
        biosphere_db_name: str = "ecoinvent-3.11-biosphere",
        mass_strategy: str = "full_mass",
        energy_strategy: str = "full_energy"
    ):
        """Initialize with caching for performance."""
        super().__init__(
            project_name=project_name,
            excluded_flows=excluded_flows,
            valuable_water_compartments=valuable_water_compartments,
            exclude_water=exclude_water,
            technosphere_db_name=technosphere_db_name,
            biosphere_db_name=biosphere_db_name,
            mass_strategy=mass_strategy,
            energy_strategy=energy_strategy
        )
        self._lookup_cache = {}
        self._flow_category_cache = {}  # Cache flow categorizations

    def _get_lookups(self) -> Dict:
        """Get or build cached lookups (thread-safe for single-threaded use)."""
        if '_lookups' not in self._lookup_cache:
            logger.info("Building combined mass/energy lookup...")
            mass_lookup, water_lookup, dry_mass_lookup, energy_lookup = self.build_combined_lookup()
            
            logger.info("Building CED energy lookup...")
            ced_total_dict, ced_renewable_dict, ced_nonrenewable_dict = self.build_ced_energy_lookup()
            
            self._lookup_cache['_lookups'] = {
                'mass': mass_lookup,
                'water': water_lookup,
                'dry_mass': dry_mass_lookup,
                'energy': energy_lookup,
                'ced_total': ced_total_dict,
                'ced_renewable': ced_renewable_dict,
                'ced_nonrenewable': ced_nonrenewable_dict
            }
        return self._lookup_cache['_lookups']

    def _is_natural_resource_flow(self, flow: Dict) -> bool:
        """
        Check if flow is a natural resource (V).
        EXACTLY mirrors calculator logic.
        """
        flow_type = flow.get('type', '').lower()
        categories = flow.get('categories', ())

        if 'resource' in flow_type:
            return True

        if categories:
            for cat in categories:
                if isinstance(cat, str) and 'resource' in cat.lower():
                    return True

        return False

    def _is_valuable_water_emission(self, flow: Dict) -> bool:
        """
        Check if flow is a valuable water emission (Er).
        EXACTLY mirrors calculator logic.
        """
        name = flow.get('name', '')
        categories = flow.get('categories', ())

        if name != 'Water':
            return False

        if not isinstance(categories, (list, tuple)) or len(categories) < 2:
            return False

        if not isinstance(categories[0], str) or categories[0].lower() != 'water':
            return False

        # Check ALL categories (not just categories[1:]) to match calculator
        return any(
            isinstance(cat, str) and cat.lower().startswith(vwc.lower())
            for cat in categories
            for vwc in self.VALUABLE_WATER_COMPARTMENTS
        )

    def _is_excluded_emission(self, flow: Dict) -> bool:
        """
        Check if flow is an excluded emission.
        EXACTLY mirrors calculator logic.
        """
        unit = flow.get('unit', '')
        name = flow.get('name', '')

        if unit != 'kilogram':
            return False

        return name in self.EXCLUDED_FLOWS

    def categorize_flow(self, flow: Dict) -> Optional[str]:
        """
        Categorize a flow according to circularity logic.
        Results cached for performance.

        Returns one of: 'V', 'technosphere', 'Er', 'W', or None (excluded)

        Note: 'technosphere' encompasses both Ri and Rr.
        The distinction happens at LCI level:
        - Ri: positive inventory amounts × positive CF = positive contribution
        - Rr: negative inventory amounts × positive CF = negative contribution
        """
        # Check cache first
        # Try to use an explicit 'key' if present, otherwise build a stable composite key
        flow_key = None
        if isinstance(flow, dict):
            flow_key = flow.get('key')
        else:
            flow_key = getattr(flow, 'key', None)

        if flow_key is None:
            categories = flow.get('categories', ()) if isinstance(flow, dict) else getattr(flow, 'categories', ())
            flow_key = f"{flow.get('name', '')}|{flow.get('unit', '')}|{flow.get('type', '')}|{tuple(categories)}"
        if flow_key in self._flow_category_cache:
            return self._flow_category_cache[flow_key]

        unit = flow.get('unit', '')
        name = flow.get('name', '')
        categories = flow.get('categories', ())
        flow_type = flow.get('type', '').lower()
        # categories = flow.get('categories', ())
        # if isinstance(categories, str):
        #     try:
        #         categories = ast.literal_eval(categories)
        #     except Exception:
        #         categories = ()
        # flow_type = flow.get('type', '').lower()

        # if self.should_exclude_flow(unit, name, categories, flow_type):

        # Check exclusion first (exactly as calculator does)
        if self.should_exclude_flow(unit, name, categories, flow_type):
            self._flow_category_cache[flow_key] = None
            return None

        # Natural resource (V)
        if self._is_natural_resource_flow(flow):
            self._flow_category_cache[flow_key] = 'V'
            return 'V'

        # Technosphere flows - will be Ri or Rr depending on LCI sign
        if flow_type == 'technosphere':
            self._flow_category_cache[flow_key] = 'technosphere'
            return 'technosphere'

        # Emissions
        if flow_type == 'emission':
            if self._is_valuable_water_emission(flow):
                self._flow_category_cache[flow_key] = 'Er'
                return 'Er'
            elif self._is_excluded_emission(flow):
                self._flow_category_cache[flow_key] = None
                return None
            else:
                self._flow_category_cache[flow_key] = 'W'
                return 'W'

        self._flow_category_cache[flow_key] = None
        return None

    def get_cf_for_flow(
        self, 
        flow: Dict, 
        strategy: str = "mass",
        mass_strategy: Optional[str] = None,
        energy_strategy: Optional[str] = None
    ) -> Optional[float]:
        """
        Get the base characterization factor for a flow.

        Args:
            flow: The biosphere flow
            strategy: 'mass' or 'energy'
            mass_strategy: Override mass strategy (uses self.mass_strategy if None)
            energy_strategy: Override energy strategy (uses self.energy_strategy if None)

        Returns:
            Base CF value or None
        """
        lookups = self._get_lookups()

        # Get all property factors (same call as calculator)
        kg_factor, kg_water_factor, kg_dry_mass_factor, kg_dry_mass_factor_2, \
            MJ_factor, MJ_renewable_factor, MJ_nonrenewable_factor = self.get_property_factors(
                flow, 
                lookups['mass'], 
                lookups['water'], 
                lookups['dry_mass'], 
                lookups['energy'],
                lookups['ced_total'], 
                lookups['ced_renewable'], 
                lookups['ced_nonrenewable']
            )

        if strategy == "mass":
            ms = mass_strategy or self.mass_strategy
            
            if ms == "full_mass":
                return kg_factor
            elif ms == "water_mass":
                return kg_water_factor
            elif ms == "dry_mass":
                # Prefer explicit dry mass, fall back to computed
                return kg_dry_mass_factor if kg_dry_mass_factor is not None else kg_dry_mass_factor_2
            elif ms == "dry_mass_2":
                return kg_dry_mass_factor_2
            else:
                raise ValueError(f"Unknown mass strategy: {ms}")

        elif strategy == "energy":
            es = energy_strategy or self.energy_strategy
            
            if es == "full_energy":
                return MJ_factor
            elif es == "renewable_energy":
                return MJ_renewable_factor
            elif es == "non_renewable_energy":
                return MJ_nonrenewable_factor
            else:
                raise ValueError(f"Unknown energy strategy: {es}")
        
        return None

    def build_cf_list(
        self,
        method_key: str,
        mass_strategy: Optional[str] = None,
        energy_strategy: Optional[str] = None,
        include_water: bool = True
    ) -> List[Tuple]:
        """
        Build characterization factor list for a specific method type.

        Args:
            method_key: Key from METHOD_TYPES ('V', 'V+R', 'W', 'V+R+Er')
            mass_strategy: Mass strategy to use
            energy_strategy: Energy strategy to use
            include_water: Whether to include water flows (m³ water flows)

        Returns:
            List of (flow_key, cf) tuples
        """
        if method_key not in self.METHOD_TYPES:
            raise ValueError(f"Unknown method key: {method_key}. Available: {list(self.METHOD_TYPES.keys())}")
        
        method_def = self.METHOD_TYPES[method_key]
        cf_list = []
        
        db = bd.Database(self.biosphere_db_name)
        total_flows = len(db)
        
        # Determine strategy
        is_energy = energy_strategy is not None
        strategy = "energy" if is_energy else "mass"
        
        logger.info(f"Building CFs for {method_def['name']} (strategy: {strategy})")
        
        for i, flow in enumerate(db):
            if i % 1000 == 0:
                logger.info(f"  Processing flow {i}/{total_flows}")
            
            # Handle water flow exclusion
            if not include_water and self.EXCLUDE_WATER:
                unit = flow.get('unit', '').lower()
                name = flow.get('name', '').lower()
                if unit == 'cubic meter' and 'water' in name:
                    continue
            
            # Categorize the flow
            category = self.categorize_flow(flow)
            
            # Skip if not in this method's categories
            if category is None or category not in method_def['categories']:
                continue
            
            # Get base CF
            base_cf = self.get_cf_for_flow(
                flow, 
                strategy=strategy,
                mass_strategy=mass_strategy,
                energy_strategy=energy_strategy
            )
            
            if base_cf is None or base_cf == 0:
                continue
            
            # Apply sign based on method definition
            sign = method_def['cf_signs'].get(category, 1)
            cf_val = sign * abs(base_cf)
            
            if cf_val != 0:
                # Use the flow's explicit key when available, otherwise mirror the composite used above
                fk = flow.get('key') if isinstance(flow, dict) else getattr(flow, 'key', None)
                if fk is None:
                    categories = flow.get('categories', ()) if isinstance(flow, dict) else getattr(flow, 'categories', ())
                    fk = f"{flow.get('name', '')}|{flow.get('unit', '')}|{flow.get('type', '')}|{tuple(categories)}"
                cf_list.append((fk, cf_val))
        
        logger.info(f"  Generated {len(cf_list)} CFs for {method_def['name']}")
        return cf_list

    def _get_strategy_suffix(
        self,
        mass_strategy: Optional[str] = None,
        energy_strategy: Optional[str] = None,
        include_water: bool = True
    ) -> str:
        """Generate strategy suffix for method names."""
        if energy_strategy:
            strategy_map = {
                "full_energy": "fe",
                "renewable_energy": "re",
                "non_renewable_energy": "nre"
            }
            return strategy_map.get(energy_strategy, energy_strategy)
        else:
            ms = mass_strategy or self.mass_strategy
            strategy_map = {
                "full_mass": "fm",
                "water_mass": "wm",
                "dry_mass": "dm",
                "dry_mass_2": "dm2"
            }
            suffix = strategy_map.get(ms, ms)
            water_suffix = "no-w" if not include_water else "all"
            return f"{suffix}_{water_suffix}"

    def create_method(
        self,
        method_key: str,
        mass_strategy: Optional[str] = None,
        energy_strategy: Optional[str] = None,
        include_water: bool = True
    ) -> Optional[bd.Method]:
        """
        Create a single LCIA method.

        Args:
            method_key: Key from METHOD_TYPES
            mass_strategy: Mass strategy (for mass methods)
            energy_strategy: Energy strategy (for energy methods)
            include_water: Include water flows

        Returns:
            Method object or None
        """
        method_def = self.METHOD_TYPES[method_key]
        
        # Determine flow type (mass or energy)
        is_energy = energy_strategy is not None
        flow_type = "Energy Flows" if is_energy else "Mass Flows"
        
        # Build strategy suffix
        strategy_suffix = self._get_strategy_suffix(mass_strategy, energy_strategy, include_water)
        
        # Build method name tuple
        method_name = (
            "Circularity Indicator",
            flow_type,
            f"{method_def['name']} ({strategy_suffix})"
        )
        
        # Build description
        desc_parts = [method_def['description']]
        if is_energy:
            desc_parts.append(f"Energy strategy: {energy_strategy}")
        else:
            desc_parts.append(f"Mass strategy: {mass_strategy or self.mass_strategy}")
            desc_parts.append(f"Water flows: {'included' if include_water else 'excluded'}")
        
        description = ". ".join(desc_parts)
        
        # Set unit
        unit = "MJ-eq" if is_energy else "kg-eq"
        
        # Temporarily set strategies for the build
        orig_mass, orig_energy = self.mass_strategy, self.energy_strategy
        try:
            if mass_strategy:
                self.mass_strategy = mass_strategy
            if energy_strategy:
                self.energy_strategy = energy_strategy
            
            # Build CF list
            cf_list = self.build_cf_list(
                method_key=method_key,
                mass_strategy=mass_strategy,
                energy_strategy=energy_strategy,
                include_water=include_water
            )
            
            # Register and write method
            method = bd.Method(method_name)
            metadata = {
                'unit': unit,
                'description': description,
                'source': 'circularity_lci',
                'version': '1.0',
                'num_cfs': len(cf_list),
                'application': 'Circularity Assessment',
                'method_key': method_key,
                'mass_strategy': mass_strategy,
                'energy_strategy': energy_strategy,
                'include_water': include_water
            }
            
            method.register(**metadata)
            method.write(cf_list)
            
            logger.info(f"✓ Created: {'::'.join(method_name)} ({len(cf_list)} CFs)")
            return method
            
        except Exception as e:
            logger.error(f"✗ Failed: {method_name}: {e}")
            return None
        finally:
            self.mass_strategy, self.energy_strategy = orig_mass, orig_energy

    def create_all_mass_methods(self, include_water: bool = True) -> Dict[str, Optional[bd.Method]]:
        """
        Create all mass-based LCIA methods.
        
        Creates 4 method types × 4 strategies = 16 methods
        (or 4 × 4 × 2 = 32 with water on/off)

        Args:
            include_water: Whether to include water flows

        Returns:
            Dictionary of method identifiers to Method objects
        """
        methods = {}
        water_label = "with_water" if include_water else "without_water"
        
        logger.info(f"Creating mass methods ({water_label})...")
        
        for method_key in self.METHOD_TYPES:
            for mass_strategy in self.MASS_STRATEGIES:
                method = self.create_method(
                    method_key=method_key,
                    mass_strategy=mass_strategy,
                    include_water=include_water
                )
                key = f"{method_key}_{mass_strategy}_{water_label}"
                methods[key] = method
        
        return methods

    def create_all_energy_methods(self) -> Dict[str, Optional[bd.Method]]:
        """
        Create all energy-based LCIA methods.
        
        Creates 4 method types × 3 strategies = 12 methods

        Returns:
            Dictionary of method identifiers to Method objects
        """
        methods = {}
        
        logger.info("Creating energy methods...")
        
        for method_key in self.METHOD_TYPES:
            for energy_strategy in self.ENERGY_STRATEGIES:
                method = self.create_method(
                    method_key=method_key,
                    energy_strategy=energy_strategy
                )
                key = f"{method_key}_{energy_strategy}"
                methods[key] = method
        
        return methods

    def create_all_methods(self) -> Dict[str, Optional[bd.Method]]:
        """
        Create ALL LCIA methods:
        - 16 mass methods with water (4 types × 4 strategies)
        - 16 mass methods without water (4 types × 4 strategies)  
        - 12 energy methods (4 types × 3 strategies)
        
        Total: 66 methods

        Returns:
            Dictionary of all method identifiers to Method objects
        """
        all_methods = {}
        
        logger.info("=" * 60)
        logger.info("Creating all circularity LCIA methods")
        logger.info("=" * 60)
        
        # Mass methods with water
        logger.info("\n[1/3] Mass methods WITH water...")
        all_methods.update(self.create_all_mass_methods(include_water=True))
        
        # Mass methods without water
        logger.info("\n[2/3] Mass methods WITHOUT water...")
        all_methods.update(self.create_all_mass_methods(include_water=False))
        
        # Energy methods
        logger.info("\n[3/3] Energy methods...")
        all_methods.update(self.create_all_energy_methods())
        
        # Summary
        successful = sum(1 for m in all_methods.values() if m is not None)
        logger.info("\n" + "=" * 60)
        logger.info(f"Created {successful}/{len(all_methods)} methods successfully")
        logger.info("=" * 60)
        
        return all_methods

    def get_method_names(self) -> Dict[str, List[str]]:
        """
        Get organized lists of created method names for easy selection.

        Returns:
            Dictionary with method names organized by category
        """
        names = {
            "mass_with_water": [],
            "mass_without_water": [],
            "energy": []
        }
        
        for method_key in self.METHOD_TYPES:
            method_def = self.METHOD_TYPES[method_key]
            
            # Mass with water
            for ms in self.MASS_STRATEGIES:
                suffix = self._get_strategy_suffix(mass_strategy=ms, include_water=True)
                names["mass_with_water"].append(
                    f"Circularity Indicator::Mass Flows::{method_def['name']} ({suffix})"
                )
            
            # Mass without water
            for ms in self.MASS_STRATEGIES:
                suffix = self._get_strategy_suffix(mass_strategy=ms, include_water=False)
                names["mass_without_water"].append(
                    f"Circularity Indicator::Mass Flows::{method_def['name']} ({suffix})"
                )
            
            # Energy
            for es in self.ENERGY_STRATEGIES:
                suffix = self._get_strategy_suffix(energy_strategy=es)
                names["energy"].append(
                    f"Circularity Indicator::Energy Flows::{method_def['name']} ({suffix})"
                )
        
        return names


def create_circularity_lcia_methods(
    project_name: str,
    excluded_flows: List[str],
    valuable_water_compartments: List[str],
    exclude_water: bool = True,
    technosphere_db_name: str = "ecoinvent-3.11-cutoff",
    biosphere_db_name: str = "ecoinvent-3.11-biosphere",
    mass_strategy: str = "full_mass",
    energy_strategy: str = "full_energy"
) -> LCIAMethodBuilder:
    """
    Factory function to create LCIA method builder.

    Example usage:
    ```python
    builder = create_circularity_lcia_methods(
        project_name="my_project",
        excluded_flows=["BOD", "COD"],
        valuable_water_compartments=["groundwater", "lake"],
        exclude_water=True
    )
    
    # Create specific methods
    mass_methods = builder.create_all_mass_methods(include_water=True)
    
    # Or create everything
    all_methods = builder.create_all_methods()
    
    # Get organized method names for selection
    names = builder.get_method_names()
    """
    return LCIAMethodBuilder(
        project_name=project_name,
        excluded_flows=excluded_flows,
        valuable_water_compartments=valuable_water_compartments,
        exclude_water=exclude_water,
        technosphere_db_name=technosphere_db_name,
        biosphere_db_name=biosphere_db_name,
        mass_strategy=mass_strategy,
        energy_strategy=energy_strategy,
    )