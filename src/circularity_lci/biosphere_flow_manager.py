import time
import logging
import bw2data as bd  # type: ignore[import]
from bw2data import utils as bw2data_utils  # type: ignore[import]

logger = logging.getLogger(__name__)


class BiosphereFlowManager:
    """
    Manage the product flows that you want to characterize, therefore that should be duplicated as a biosphere flow.
    """

    def __init__(self, biosphere_db_name):
        self.biosphere_db_name = biosphere_db_name
        self.biosphere_db = bd.Database(biosphere_db_name)

    def create_or_get_biosphere_flow(self, activity, production_exchange):
        """Create or reuse a biosphere flow for a burden-free activity"""
        # ref_name = production_exchange['name'] 23/04
        # ref_unit = production_exchange['unit'] 23/04
        # --- Name hierarchy ---
        ref_name = production_exchange.get('name')
        if ref_name is None:
            ref_name = activity.get('name')

        # --- Unit hierarchy ---
        ref_unit = production_exchange.get('unit')
        if ref_unit is None:
            ref_unit = activity.get('unit')

        compartment = 'technosphere'  # the flow is from the technosphere, or outside of the product system of study, the flows are from another economic environment
        subcomp = activity.get('location', 'Unknown')  # same reference flow's name can come from the same activity's name, we must distinguish them furtherly

        for flow in self.biosphere_db:
            if (flow.get('name') == ref_name and
                flow.get('unit') == ref_unit and
                flow.get('categories') == (compartment, subcomp)):
                return flow.key, False

        new_code = str(activity.key) + "_BF_" + bw2data_utils.random_string(5)  # can be changed to IF (Intermediate flow) or PF (Product Flow)
        new_flow = self.biosphere_db.new_activity(code=new_code)
        new_flow.update({
            'database': self.biosphere_db_name,
            'code': new_code,
            'name': ref_name,
            'unit': ref_unit,
            'categories': (compartment, subcomp),
            'type': 'technosphere'  # not a natural resource, not a emission flow, a technosphere flow. Can be change according to a better nomenclature e.g.: "economical" could fit.
        })
        new_flow.save()
        return new_flow.key, True
    
    # Improved, for later
    def get_exchange_property(exc, act, key, fallback_key=None, default=None):
        """Generic safe getter with hierarchy: exchange → activity → default"""
        value = exc.get(key) # type: ignore
        if value is not None:
            return value
        if fallback_key:
            return act.get(fallback_key, default)
        return act.get(key, default)
        # amount = get_exchange_property(prod, act, 'amount', default=1) # in prepare_exchanges()
        # unit = get_exchange_property(prod, act, 'unit') # in create_or_get_biosphere_flow()
        # ref_name = get_exchange_property(prod, act, 'name') # in create_or_get_biosphere_flow()

    def prepare_exchanges(self, burden_free_activities):
        """Prepare biosphere exchanges"""
        new_exchanges = []
        flows_created = 0

        for db_name, activities in burden_free_activities.items():
            logger.info("Processing '%s':", db_name)
            for act in activities:

                prods = list(act.production())
                if not prods:
                    logger.warning("No production exchange for %s", act)
                    continue
                prod = list(act.production())[0]
                # amount, unit = prod['amount'], prod['unit'] 23/04
                # --- Amount hierarchy ---
                amount = prod.get('amount')
                if amount is None:
                    amount = act.get('production amount', 1)  # fallback, rarely needed 
                # --- Unit hierarchy ---
                unit = prod.get('unit')
                if unit is None:
                    unit = act.get('unit')
                flow_key, created = self.create_or_get_biosphere_flow(act, prod)
                if created:
                    flows_created += 1

                new_exchanges.append((act.key, {
                    'input': flow_key,
                    'amount': amount,
                    'unit': unit,
                    'type': 'biosphere',
                    'output': act.key
                }))
                logger.debug("Prepared exchange for %s", act.get('name'))
        return new_exchanges, flows_created

    def add_exchanges(self, new_exchanges):
        """Add biosphere exchanges to burden-free activities"""
        for act_key, exc in new_exchanges:
            act = bw2data_utils.get_activity(act_key)
            if any(e['input'] == exc['input'] and e['type'] == exc['type'] for e in act.exchanges()):
                continue
            new_exc = act.new_exchange(**exc)
            new_exc.save()

    def process(self, burden_free_activities):
        """End-to-end biosphere flow creation"""
        logger.info("Step 2: Creating Biosphere Flows")
        start = time.time()

        new_exchanges, flows_created = self.prepare_exchanges(burden_free_activities)
        logger.info("Created %d new biosphere flows in '%s'.", flows_created, self.biosphere_db.name)

        self.add_exchanges(new_exchanges)

        logger.info("Step 2 completed in %.2f s", time.time() - start)