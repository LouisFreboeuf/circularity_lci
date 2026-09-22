import ast
import logging
import traceback

import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import bw2data as bd # type: ignore
import bw2calc as bc # type: ignore

logger = logging.getLogger(__name__)


class MultiLCACalculator:
    """
    Enhanced class to create and run MultiLCA calculations with circularity indicators.
    Supports multiple activities, custom amounts, and manual setup creation.
    """

    def __init__(self, technosphere_db_name):
        self.location_lookup = {}
        self.technosphere_db_name = technosphere_db_name


    def get_circularity_methods(self):
        """Get all circularity indicator methods."""
        return [
            m 
            for m in bd.methods 
            if m and len(m) > 0 and str(m[0]).lower().startswith('circularity')]

    def precompute_locations(self):
        """Precompute locations for all activities in the database."""
        db = bd.Database(self.technosphere_db_name) #bw. for brightway2
        self.location_lookup = {
            activity.key: activity.get('location', 'Unknown')
            for activity in db
        }
        return self.location_lookup

    def find_activities(self, search_terms, location_codes=None):
        """
        Find activities matching search terms and location codes.
        search_terms can be a list of multiple reference products to search for.
        """
        if not self.location_lookup:
            self.precompute_locations() # Before, an argument in precompute_locations: self.technosphere_db_name

        db = bd.Database(self.technosphere_db_name) 
        location_codes = [code.lower() for code in (location_codes or [])]
        search_terms = [term.lower() for term in search_terms]

        matches = []
        for act in db:
            name = act.get('name', '').lower()
            ref_prod = act.get('reference product', '').lower()
            unit = act.get('unit', '').lower()
            location = self.location_lookup.get(act.key, "").lower()

            # Check if any search term matches name or reference product
            term_match = any(term in name or term in ref_prod for term in search_terms)
            location_match = not location_codes or any(code in location for code in location_codes)

            if term_match and location_match:
                matches.append({
                    'key': act.key,
                    'name': act.get('name', ''),
                    'reference_product': act.get('reference product', ''),
                    'unit': act.get('unit', ''),
                    'location': self.location_lookup.get(act.key, 'Unknown')
                })

        if not matches:
            logger.info("No activities found. Here are some similar activities:")
            similar = []
            for act in db:
                name = act.get('name', '').lower()
                ref_prod = act.get('reference product', '').lower()
                if any(term in name or term in ref_prod for term in search_terms):
                    similar.append({
                        'name': act.get('name', ''),
                        'reference_product': act.get('reference product', ''),
                        'unit': act.get('unit', ''),
                        'location': self.location_lookup.get(act.key, 'Unknown')
                    })
                    if len(similar) >= 10:
                        break

            if similar:
                df = pd.DataFrame(similar)
                logger.info(df.to_string(index=False))
            raise ValueError("No matching activities found")

        return pd.DataFrame(matches)

    def select_activities(self):
        """
        Interactive activity selection with support for multiple activities and custom amounts.
        Returns a list of (activity_key, amount) tuples.
        """
        print("\n=== Activity Selection ===")

        # Get search parameters with better instructions
        print("Enter reference products to search for (comma-separated):")
        print("Example: 'electricity, high voltage', 'steel', 'concrete'")
        search_terms = input("> ").strip().split(',')
        search_terms = [term.strip() for term in search_terms if term.strip()]

        location_input = input("Enter location codes (comma-separated, or leave empty): ").strip()
        location_codes = [code.strip().lower() for code in location_input.split(',')] if location_input else None

        # Find matching activities
        df = self.find_activities(search_terms, location_codes)

        print(f"\nFound {len(df)} matching activities:")
        print(df.to_string(index=True))

        # Let user select activities
        selected_activities = []
        while True:
            try:
                selection = input("\nEnter activity indices to add (comma-separated), or 'done' to finish: ")
                if selection.lower() == 'done':
                    if not selected_activities:
                        print("You must select at least one activity.")
                        continue
                    break

                indices = [int(idx.strip()) for idx in selection.split(',')]
                for idx in indices:
                    if 0 <= idx < len(df):
                        activity = df.iloc[idx]
                        print(f"\nSelected: {activity['reference_product']} ({activity['unit']})")

                        # Get amount for this activity
                        while True:
                            try:
                                amount = float(input(f"Enter amount for this activity (unit: {activity['unit']}): "))
                                if amount <= 0:
                                    print("Amount must be positive.")
                                    continue
                                selected_activities.append((activity['key'], amount))
                                break
                            except ValueError:
                                print("Please enter a valid number.")
                    else:
                        print(f"Index {idx} is out of range (0-{len(df)-1}).")
            except ValueError:
                print("Please enter valid indices or 'done'.")

        return selected_activities

    def create_calculation_setup_interactive(self, setup_name):
        """
        Create a calculation setup interactively with multiple activities.
        """
        print(f"\n=== Creating Calculation Setup: {setup_name} ===")

        # Select activities
        functional_units = self.select_activities()

        # Get circularity methods
        circ_methods = self.get_circularity_methods()

        if not circ_methods:
            raise ValueError("No circularity methods found. Available methods should start with 'Circularity'")

        print(f"\nFound {len(circ_methods)} circularity methods:")
        for i, method in enumerate(circ_methods[:5]):  # Show first 5 as examples
            print(f"{i+1}. {' | '.join(str(m) for m in method)}")
        if len(circ_methods) > 5:
            print(f"... and {len(circ_methods)-5} more")

        # Create calculation setup
        bd.calculation_setups[setup_name] = {
            "inv": [dict([fu]) for fu in functional_units],
            "ia": circ_methods,
            "description": f"MultiLCA with {len(functional_units)} functional units and {len(circ_methods)} circularity methods"
        }

        print(f"\nCalculation setup '{setup_name}' created successfully!")
        print(f"- Functional units: {len(functional_units)}")
        print(f"- Methods: {len(circ_methods)} circularity indicators")
        for i, (key, amount) in enumerate(functional_units):
            act = bd.get_activity(key)
            print(f"  {i+1}. {act.get('reference product')} ({act.get('unit')}): {amount}")

        return bd.calculation_setups[setup_name]

    def create_calculation_setup_manual(self, setup_name):
        """
        Create a calculation setup manually by entering JSON-like data.
        """
        print(f"\n=== Manual Calculation Setup Creation: {setup_name} ===")
        print("Enter your functional units in the format: activity_key:amount (one per line)")
        print("Example: ('database', 'key'):1.5")
        print("Enter 'done' when finished.")

        functional_units = []
        while True:
            fu_input = input(f"Functional unit {len(functional_units)+1}: ").strip()
            if fu_input.lower() == 'done':
                if not functional_units:
                    print("You must enter at least one functional unit.")
                    continue
                break
            try:
                # Parse the input
                key_str, amount_str = fu_input.split(':')
                # key = eval(key_str.strip())  # Convert string to tuple
                key = ast.literal_eval(key_str.strip()) # safer parsing to tuple
                amount = float(amount_str.strip())

                # Validate the activity exists
                bd.get_activity(key)
                functional_units.append((key, amount))
            except Exception as e:
                print(f"Invalid input: {str(e)}. Please try again.")

        # Get circularity methods
        circ_methods = self.get_circularity_methods()
        if not circ_methods:
            raise ValueError("No circularity methods found")

        # Create calculation setup
        bd.calculation_setups[setup_name] = {
            "inv": [dict([fu]) for fu in functional_units],
            "ia": circ_methods,
            "description": f"Manual MultiLCA setup with {len(functional_units)} functional units"
        }

        print(f"\nCalculation setup '{setup_name}' created successfully!")
        print(f"- Functional units: {len(functional_units)}")
        print(f"- Methods: {len(circ_methods)} circularity indicators")
        for i, (key, amount) in enumerate(functional_units):
            act = bd.get_activity(key)
            print(f"  {i+1}. {act.get('reference product')} ({act.get('unit')}): {amount}")

        return bd.calculation_setups[setup_name]

    def run_multilca(self, setup_name):
        """
        Run MultiLCA for a given setup and return results.
        """
        if setup_name not in bd.calculation_setups:
            raise ValueError(f"Calculation setup '{setup_name}' not found. Available setups: {list(bd.calculation_setups.keys())}")

        # Create MultiLCA object
        mLCA = bc.MultiLCA(setup_name) #bw. for brightway2

        # Create results DataFrame
        calculation_setup = bd.calculation_setups[setup_name]

        columns = []
        for fu_item in calculation_setup['inv']:
            for activity_key, amount in fu_item.items():
                activity = bd.get_activity(activity_key)
                columns.append(f"{activity['name']} ({activity['location']}) - {amount} {activity.get('unit', 'unit')}")

        method_names = []
        for method in calculation_setup['ia']:
            if isinstance(method, tuple):
                method_names.append(" | ".join(str(m) for m in method))
            else:
                method_names.append(str(method))

        # Create and return the DataFrame
        mLCAdf = pd.DataFrame(
            index=method_names,
            columns=columns,
            data=mLCA.results.T
        )

        return mLCAdf

    def plot_results2(self, results_df, title=None):
        """
        Plot the MultiLCA results DataFrame grouped by LCIA categories with units.
        """
        if not isinstance(results_df, pd.DataFrame):
            raise ValueError("Input must be a DataFrame from run_multilca()")

        # Group methods by category
        material_with_water = []
        material_without_water = []
        energy = []

        # Extract units from method names
        units = {}

        for method in results_df.index:
            if isinstance(method, tuple):
                # Method is in tuple format: ("Circularity Indicator", "Material flows", "Mass - Virgin Resource Input", "kg-eq")
                if len(method) >= 4:
                    category = method[1]  # Second element is category (Material flows/Energy flows)
                    method_name = method[2]  # Third element is method name
                    unit = method[3]  # Fourth element is unit (kg-eq/MJ-eq)

                    units[method] = unit

                    if "Material flows" in category:
                        if "Without Water" in method_name:
                            material_without_water.append(method)
                        else:
                            material_with_water.append(method)
                    elif "Energy flows" in category:
                        energy.append(method)
            elif isinstance(method, str):
                # Fallback for string methods (less likely in your case)
                method_parts = method.split(" | ")
                if len(method_parts) >= 3:
                    category = method_parts[1]
                    method_name = method_parts[2]

                    # Try to extract unit from method name
                    if "kg-eq" in method_name:
                        unit = "kg-eq"
                    elif "MJ-eq" in method_name:
                        unit = "MJ-eq"
                    else:
                        unit = "unit"

                    units[method] = unit

                    if "Material flows" in category:
                        if "Without Water" in method_name:
                            material_without_water.append(method)
                        else:
                            material_with_water.append(method)
                    elif "Energy flows" in category:
                        energy.append(method)

        # Create a figure with subplots for each category
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 18))
        if title:
            fig.suptitle(title, y=1.02)
        else:
            fig.suptitle("MultiLCA Results by Category", y=1.02)

        # Plot each category
        categories = [
            ("Material Flows (with water)", material_with_water, axes[0]),
            ("Material Flows (without water)", material_without_water, axes[1]),
            ("Energy Flows", energy, axes[2])
        ]

        for category_name, methods, ax in categories:
            if methods:
                # Filter dataframe for this category
                category_df = results_df.loc[methods]

                # Plot
                category_df.plot(kind='bar', ax=ax, legend=False)

                # Formatting
                ax.set_title(category_name)

                # Get the unit for this category
                if methods:
                    sample_method = methods[0]
                    unit = units.get(sample_method, "unit")
                    ax.set_ylabel(f"Impact Score ({unit})")

                ax.grid(axis='y', alpha=0.3)

                # Rotate x-axis labels
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            else:
                ax.axis('off')
                ax.set_title(f"No {category_name} methods found")

        plt.tight_layout()
        plt.show()

    def plot_results(self, results_df, title=None):
        """Plot the MultiLCA results DataFrame."""
        plt.figure(figsize=(12, 8))
        results_df.plot(kind='bar', ax=plt.gca())

        if title:
            plt.title(title)
        else:
            plt.title("MultiLCA Results")

        plt.ylabel("Impact Score")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        plt.show()

    def full_workflow(self):
        """
        Complete workflow from setup creation to results visualization.
        """
        print("\n=== MultiLCA Calculation Workflow ===")

        try:
            # Choose creation method
            print("\nSelect setup creation method:")
            print("1. Interactive setup with activity search")
            print("2. Manual setup creation")
            choice = input("Enter your choice (1 or 2): ").strip()

            setup_name = input("\nEnter a name for this calculation setup: ").strip()

            if choice == "1":
                # Interactive setup
                self.create_calculation_setup_interactive(setup_name)
            elif choice == "2":
                # Manual setup
                self.create_calculation_setup_manual(setup_name)
            else:
                print("Invalid choice. Please enter 1 or 2.")
                return None

            # Run MultiLCA
            print("\nRunning MultiLCA...")
            results = self.run_multilca(setup_name)

            # Show and plot results
            print("\nMultiLCA Results:")
            print(results)

            plot_title = f"Results for {setup_name}"
            self.plot_results2(results, title=plot_title) # or plot_results1 for rough ploting.

            return results

        except Exception as e:
            logger.error("Error: %s", str(e))
            logger.debug("Traceback:\n%s", traceback.format_exc())
            return None