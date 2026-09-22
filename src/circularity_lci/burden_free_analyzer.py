# used for self.all_burden_free_activities (or replace with dict).
from collections import defaultdict

# used for FS, timing, caching.
import time
import os
import json
import logging 

logger = logging.getLogger(__name__)

# used for importing ecoinvent
import bw2io as bi # type: ignore

# sed for Database, get_activity, bd.databases.
import bw2data as bd # type: ignore

# package relative import
from .functions_bafu_from_sacchi import import_bafu_from_sacchi
from ._packaging import get_data_path

class BurdenFreeAnalyzer:
    """
    Main class for analyzing burden-free activities and modifying the intervention matrix.
    This class can be modified in function of the technosphere activities to be characterized.
    """

    def __init__(self, project_name, database_provider, ecospold_folder, technosphere_db_name, database_version, database_systemmodel):
        self.project_name = project_name
        self.database_provider = database_provider
        self.ecospold_folder = ecospold_folder # if required for bafu or ei version 2
        self.technosphere_db_name = technosphere_db_name
        self.database_version = database_version
        self.database_systemmodel = database_systemmodel
        self.all_burden_free_activities = defaultdict(list)
        self.setup_complete = False

    def setup_project(self):
        """Set up a new Brightway2 project and import ecoinvent if missing"""
        if self.project_name not in bd.projects:
            bd.projects.create_project(self.project_name)
       
        bd.projects.set_current(self.project_name)
        logger.info("The project '%s' is set as current.", self.project_name)
        # the principles are applied to the most used LCA database, ecoinvent [https://ecochain.com/blog/lci-databases-in-lca/]
        # the cutoff system model is the simplest to understand [https://support.ecoinvent.org/system-models-1]
        if self.database_provider == "ecoinvent":
            if self.technosphere_db_name not in bd.databases: 
                logging.info("Importing the specified technosphere LCA database (ecoinvent)...")
                # For ecoinvent
                username = os.getenv("EI_USERNAME")
                password = os.getenv("EI_PASSWORD")
                bi.import_ecoinvent_release(self.database_version, self.database_systemmodel, username, password) # change the version, and system model
            
            elif self.database_provider == "bafu":
                logging.info("Importing BAFU via sacchi...")
                import_bafu_from_sacchi(
                    self.ecospold_folder,
                    mapping_csv = get_data_path("elementary_flows_mapping.csv"),
                    db_name = self.technosphere_db_name
                )
        self.setup_complete = True

    def save_results(self, filename=None):
        """Save burden-free activities to a JSON file."""

        json_dir = os.path.join("results", "json")
        os.makedirs(json_dir, exist_ok=True)

        if filename is None:
            filename = f"{self.project_name}_added_product_flows.json"

        filepath = os.path.join(json_dir, filename)

        with open(filepath, "w") as f:
            json.dump(
                {db_name: [act.as_dict() for act in acts]
                for db_name, acts in self.all_burden_free_activities.items()},
                f,
                indent=2
            )
        
    def load_results(self, filename=None):
        """Load burden-free activities from a JSON file."""

        json_dir = os.path.join("results", "json")
        os.makedirs(json_dir, exist_ok=True)

        if filename is None:
            filename = f"{self.project_name}_added_product_flows.json"

        filepath = os.path.join(json_dir, filename)

        if not os.path.exists(filepath):
            return False

        with open(filepath, "r") as f:
            data = json.load(f)

        self.all_burden_free_activities = {}

        for db_name, acts in data.items():
            self.all_burden_free_activities[db_name] = []

            for act in acts:
                try:
                    key = (db_name, act["code"])
                    activity = bd.get_activity(key)
                    self.all_burden_free_activities[db_name].append(activity)

                except Exception as e:
                    logger.error("Error loading activity %s in %s: %s", act['code'], db_name, e)

        return True

    @staticmethod
    def is_burden_free_activity(activity):
        """Check if an activity is burden-free"""
        if len(list(activity.production())) != 1:
            return False
        if len(list(activity.technosphere())) > 0:
            return False
        if len(list(activity.biosphere())) > 0:
            return False
        return True

    def analyze_database(self, db_name):
        """Analyze one database for burden-free activities"""
        db = bd.Database(db_name)
        burden_free_activities = [act for act in db if self.is_burden_free_activity(act)]

        logger.info("Analyzing database: %s", db_name)
        logger.info("Found %d burden-free activities.", len(burden_free_activities))

        for act in burden_free_activities[:5]:
            logger.info("- %s (%s, %s)", act.get('name'), act.get('location'), act.get('unit'))
        if len(burden_free_activities) > 5:
            logger.info("...")

        return burden_free_activities
    
    def count_activities_in_json(self, filename=None):
        """
        Count the number of activities in each database in a JSON file.

        Args:
            filename (str, optional): The name of the JSON file. If None, uses the default filename.

        Returns:
            dict: A dictionary with database names as keys and activity counts as values.
        """
        json_dir = os.path.join("results", "json")
        os.makedirs(json_dir, exist_ok=True)

        if filename is None:
            filename = f"{self.project_name}_added_product_flows.json"

        filepath = os.path.join(json_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} does not exist.")

        with open(filepath, "r") as f:
            data = json.load(f)

        activity_counts = {}

        for db_name, acts in data.items():
            activity_counts[db_name] = len(acts)

        return activity_counts

    def analyze_all_databases(self):
        """Analyze all databases, with caching to avoid re-analysis."""
        if not self.setup_complete:
            self.setup_project()
        # Try to load cached results
        if self.load_results():
            logger.info("Loaded burden-free activities from cache.")
            return self.all_burden_free_activities
            
        # Otherwise, perform full analysis
        logger.info("Analyzing all databases (this may take a while)")    
        self.all_burden_free_activities = {}
        total = 0
        start = time.time()

        logger.info("Analyzing all databases")
        for db_name in bd.databases:
            activities = self.analyze_database(db_name)
            self.all_burden_free_activities[db_name] = activities
            total += len(activities)
            
        # Save results to cache
        self.save_results()
        logger.info("Summary")
        for db_name, acts in self.all_burden_free_activities.items():
            logger.info("Database '%s': %d burden-free activities", db_name, len(acts))
        logger.info("Total burden-free activities: %d", total)
        logger.info("Step 1 completed in %.2f s", time.time() - start)
        
        return self.all_burden_free_activities