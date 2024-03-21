import numpy as np
import sqlite3
import WD_Textual_References_Dataset.wikidata_utils as wdutils
import requests
import pandas as pd
import ast
import WD_Textual_References_Dataset.sample_size 

import qwikidata
from qwikidata.json_dump import WikidataJsonDump
from qwikidata.linked_data_interface import get_entity_dict_from_api
from qwikidata.sparql import (get_subclasses_of_item,
                              return_sparql_query_results)
from qwikidata.entity import WikidataItem, WikidataProperty, WikidataLexeme

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

from tqdm.auto import tqdm
tqdm.pandas()

SEED=42

from IPython.display import clear_output