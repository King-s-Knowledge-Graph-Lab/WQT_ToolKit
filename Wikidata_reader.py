import time

from qwikidata.entity import WikidataItem
from qwikidata.json_dump import WikidataJsonDump
from qwikidata.utils import dump_entities_to_json

P_OCCUPATION = "P106"
Q_POLITICIAN = "Q82955"


def has_occupation_politician(item: WikidataItem, truthy: bool = True) -> bool:
    """Return True if the Wikidata Item has occupation politician."""
    if truthy:
        claim_group = item.get_truthy_claim_group(P_OCCUPATION)
    else:
        claim_group = item.get_claim_group(P_OCCUPATION)

    occupation_qids = [
        claim.mainsnak.datavalue.value["id"]
        for claim in claim_group
        if claim.mainsnak.snaktype == "value"
    ]
    return Q_POLITICIAN in occupation_qids


# create an instance of WikidataJsonDump
wjd_dump_path = "/scratch/prj/inf_wqp/wikidata-20220103-all.json.gz"
wjd = WikidataJsonDump(wjd_dump_path)

# create an iterable of WikidataItem representing politicians
politicians = []
t1 = time.time()
for ii, entity_dict in enumerate(wjd):

    if entity_dict["type"] == "item":
        entity = WikidataItem(entity_dict)
        if has_occupation_politician(entity):
            politicians.append(entity)

    if ii % 1000 == 0:
        t2 = time.time()
        dt = t2 - t1
        print(
            "found {} politicians among {} entities [entities/s: {:.2f}]".format(
                len(politicians), ii, ii / dt
            )
        )

    if ii > 10000:
        break

def get_claim_ids_for_property(item: WikidataItem, property_id: str) -> list:
    """Return a list of claim IDs for the given property_id."""
    claim_ids = []
    claim_group = item.get_claim_group(property_id)
    for claim in claim_group:
        claim_id = claim.id  # This gets the claim's ID
        claim_ids.append(claim_id)
    return claim_ids

# Example usage for an item
claim_ids = get_claim_ids_for_property(entity, P_OCCUPATION)
print(claim_ids)


"""

# write the iterable of WikidataItem to disk as JSON
out_fname = "filtered_entities.json"
dump_entities_to_json(politicians, out_fname)
wjd_filtered = WikidataJsonDump(out_fname)

# load filtered entities and create instances of WikidataItem
for ii, entity_dict in enumerate(wjd_filtered):
    item = WikidataItem(entity_dict)

"""