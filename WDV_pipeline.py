import numpy as np
import os
import bz2
import re
import random
import json
import qwikidata
from qwikidata.json_dump import WikidataJsonDump
from qwikidata.linked_data_interface import get_entity_dict_from_api
from qwikidata.sparql import (get_subclasses_of_item,
                              return_sparql_query_results)
from collections import Counter
from qwikidata.entity import WikidataItem, WikidataProperty, WikidataLexeme
import sqlite3
from itertools import islice
import time
from pprint import pprint
import traceback
import pdb
from importlib import reload

import WDV.WikidataClaims.wikidata_utils as wdutils
import requests
import hashlib


### 1. WebNLG 
wdAPI = wdutils.CachedWikidataAPI(save_every_x_queries=10)
def get_subclasses(class_entity_id):
    sparql_query = '''
        SELECT ?item ?itemLabel 
        WHERE 
        {
          ?item wdt:P279 wd:$1.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }
    '''.replace('$1',class_entity_id)
    sparql_results = wdAPI.query_sparql_endpoint(sparql_query)
    return sparql_results#['results']['bindings']

IGNORE_LIST = {
    'Q174834': ''' This is AUTHORITY, and the reason we are excluding it is because it overlaps with politician,
    on top of escaping the idea of written work. Yes, it is written that authority exists, but this begins to enter
    classes such as Mayor of a place in France, chair of local government, etc.''',
    #'Q382617': ''' This is MAYOR OF A PLACE IN FRANCE, and the reason we are excluding this is because it has over 40
    #thousand subclasses that have little to no entities in them, so we are removing them.''',
    #'Q5663900': ''' MAYOR OF A PLACE IN SPAIN, same as above.''',
    #'Q15113603': ''' member of a municipal council in France (Q15113603), same reason as above, over 40K subclasses. ''',
    #'Q20748648': ''' Same as above, member of a municipal council in the Netherlands, over 1.6k subclasses.''',
    #'Q8054': ''' TEMPORARY, to speed up Politician'''
    'Q8054': ''' PROTEIN, which is part of the subclass tree of chemical compound. This has so many subclasses that
    the Sparql endpoint times out. Also, it would be a very specific (and massively populated) area in Wikidata which
    we intend on not covering with ChemicalCompound'''
}

#subclasses_n_threshold=1000
def get_subclass_tree(root_class_id,
                      root_class_label=None,
                      level=0,
                      verbose=False,
                      all_visited_so_far = None,
                      path_so_far = None,
                      subclasses_n_threshold=10): 
    if verbose:
        print('>',' - '*level,root_class_id,'on level',level)
    if root_class_label is None:
        root_class_label = wdAPI.get_label(root_class_id)
    root_tree = {
        'label': re.sub(r'\s+', '', root_class_label.title()),
        'entity_id': root_class_id
    }
    
    if root_class_id in IGNORE_LIST.keys():
        root_tree['subclasses'] = 'IGNORE:' + IGNORE_LIST[root_class_id]
        return root_tree

    results = get_subclasses(root_class_id)['results']['bindings']
    
    if not path_so_far:
        path_so_far = []
    else:
        if root_class_id in path_so_far:
            #pdb.set_trace()
            print('>',' - '*level,'Loop detected in class',root_class_id,'on level',level)
            root_tree['subclasses'] = 'LOOP'
            return root_tree
    path_so_far = path_so_far.copy()
    path_so_far.append(root_class_id)
    
    if not all_visited_so_far:
        all_visited_so_far = []
    else:
        if root_class_id in all_visited_so_far:
            #pdb.set_trace()
            print('>',' - '*level,'Simple repetition detected in class',root_class_id,'on level',level)
            root_tree['subclasses'] = 'REPEAT'
            return root_tree
    all_visited_so_far.append(root_class_id)
            
    if len(results) == 0:
        root_tree['subclasses'] = None
    else:
        subclasses = {}
        if len(results) > subclasses_n_threshold:
            print('>',' - '*level,'Too many subclasses in',root_class_id,'on level',level,'so limiting to',subclasses_n_threshold,'first')
        for result in results[:subclasses_n_threshold]:
            subclass_id = result['item']['value'].split('/')[-1]
            subclass_tree = get_subclass_tree(
                root_class_id = subclass_id,
                root_class_label = result['itemLabel']['value'],
                level = level + 1,
                verbose = verbose,
                all_visited_so_far = all_visited_so_far,
                path_so_far = path_so_far
            )
            subclasses[subclass_id] = subclass_tree
        root_tree['subclasses'] = subclasses
    return root_tree

""" 
classes = {
    'Seen': {
        'Airport' : 'Q1248784',
        'Astronaut' : 'Q11631',
        'Building' : 'Q41176',
        'City' : 'Q515',
        'ComicsCharacter' : 'Q1114461',
        'Food' : 'Q2095',
        'Monument' : 'Q4989906',
        'SportsTeam' : 'Q12973014',
        'University' : 'Q3918',
        'WrittenWork' : 'Q47461344'
   },
    'Unseen_WebNLG': {
        'Athlete' : 'Q2066131',
        'Artist' : 'Q483501',
        'CelestialBody' : 'Q6999',
        'MeanOfTransportation' : 'Q334166',
        'Politician' : 'Q82955'
    },
    'Unseen_New': {
        #'ScholarlyArticle': 'Q13442814', This overlaps 100% with WrittenWork
        'Taxon' : 'Q16521', # This overlapps 35% with Food, which is acceptable (?)
        'Street' : 'Q79007',
        'Painting': 'Q3305213',
        'ChemicalCompound': 'Q11173',
        'Mountain': 'Q8502' # Replacing ScholarlyArticle. 
    }
}
"""

classes = {
    'Seen': {
        'Airport' : 'Q1248784',
        'Astronaut' : 'Q11631',
   }
}

root_classes = [(class_label, classes[part][class_label]) for part in classes.keys() for class_label in classes[part]]
subclass_trees = {}

from IPython.display import clear_output
for (root_class_label, root_class_id) in root_classes:
    clear_output(wait=True)
    print('Retrieving subclass tree for %s' % root_class_label)
    subclass_trees[root_class_id] = get_subclass_tree(root_class_id, root_class_label, verbose=True)
    
wdAPI.x_queries_passed = wdAPI.save_every_x_queries
wdAPI.save_entity_cache()

with open('WebNLG_to_Wikidata_Subclass_Tree_Thr=1000.json','w+') as f:
    json.dump(subclass_trees, f, indent=4)


### 2. getting wikidata entities 
import sqlite3
import numpy as np
import pandas as pd
import json
import pdb
import ast
from importlib import reload
import WDV.WikidataClaims.wikidata_utils as wdutils
db = sqlite3.connect('wikidata_claims_refs_parsed.db')
cursor = db.cursor()
claims_columns = ['entity_id','claim_id','rank','property_id','datatype','datavalue']
# Checking first few elements
cursor.execute('select * from claims limit 15;')
head_df = pd.DataFrame(cursor.fetchall())
head_df.columns = claims_columns
# Checking total of claims
cursor.execute('select count(*) from claims')
cursor.fetchall()
# To extract the values from datavalue columns
def convert_datavalue(datavalue, datatype):
    try:
        if datavalue in ['novalue','somevalue']:
            return datavalue
        datavalue = ast.literal_eval(datavalue)
        if datatype == 'wikibase-item':
            return datavalue['value']['id']
        else:
            raise Exception
    except ValueError as e:
        pdb.set_trace()
        raise e
cursor.execute('select count(*) from claims where property_id="P31"')
cursor.fetchall()
# Number of single P31 properties in the database: 19791222
cursor.execute('select claim_rank, count(claim_rank) from claims where property_id="P31" group by claim_rank')
cursor.fetchall()
# This means that 3.34% are deprecated, 96.6% are normal, and 0.07% are preferred.
# Just taking a look at the deprecated ones to see what's up
cursor.execute(
    '''
    select *
    from claims
    where property_id="P31" and claim_rank="deprecated"
    '''
)
deprecated_P31_df = pd.DataFrame(cursor.fetchall())
deprecated_P31_df.columns = claims_columns
deprecated_P31_df['class_entity'] = deprecated_P31_df.apply(
        lambda x : convert_datavalue(x['datavalue'], x['datatype']) , axis=1
)
'''
We can see that over 87% of them are just two classes: infrared source (Q67206691) and star (Q523)
The following 10 are: 
- near-IR source (Q67206785), a subclass of infrared source
- astronomical radio source (Q1931185) 
- galaxy (Q318)
- high proper-motion star (Q2247863), a subclass of star
- double star (Q13890), NOT a subclass of star  
- active galactic nucleus (Q46587), a subclass of galaxy      
- variable star (Q6243), a subclass of star      
- astrophysical X-ray source (Q2154519), NOT a subclass of infrared source 
- long period variable (Q1153690), a subclass of variable star, which is a subclass of star
- quasar (Q83373), a subclass of active galactic nucleus

This sums to 94.44% of the deprecated cases, and they are all on the domain of astronomy.
'''

deprecated_P31_df.groupby('class_entity').count().entity_id.sort_values(ascending=False).head(20)/661591*100
cursor.execute(
    '''
    select *
    from claims
    where property_id="P31" and claim_rank!="deprecated"
    '''
)
P31_df = pd.DataFrame(cursor.fetchall())
P31_df.columns = claims_columns
P31_df['class_entity'] = P31_df.apply(
        lambda x : convert_datavalue(x['datavalue'], x['datatype']) , axis=1
)
P31_df.read_csv('P31_df.csv')

class_entity_count = P31_df[['class_entity','entity_id']].groupby('class_entity').count()#.sort_values(ascending=False)
class_entity_count.columns = ['count']
class_entity_count = class_entity_count.sort_values(by='count', ascending=False).reset_index()
# Getting all classes summing up to 95% of the total amount of P31 property uses
# This is because we cant use the Wikidata API for ALL of these, so we'll only use for the most prevalent
# This also possibly cuts fringe cases and possible wrong usages of entities as classes when they should not

total_sum = class_entity_count.sum()['count']
thr = 95
for i in range(class_entity_count.shape[0]):
    partial_sum = class_entity_count.iloc[:i].sum()['count']
    if 100*partial_sum/total_sum >= thr:
        print('The %dth percentile is at the %dth row' % (thr, i))
        break
class_entity_count_95th = class_entity_count.loc[:572]
wdAPI = wdutils.CachedWikidataAPI()
class_entity_count_95th['class_entity_label'] = class_entity_count_95th['class_entity'].apply(wdAPI.get_label)
class_entity_count_95th.to_csv('class_entity_count_95th.csv', index=None)
# This listing was then used to decide on the most numerous classes we could use that are not in WebNLG
# We obtain their subclass structure with another notebook: WebNLG_Classes_To_Wikidata_Mapping
# This reveals to us all subclasses in the subclass tree of the classes from WebNLG and the ones selected from here
import json
import pandas as pd
P31_df = pd.read_csv('P31_df.csv')
class_entity_count_95th = pd.read_csv('class_entity_count_95th.csv')
with open('WebNLG_to_Wikidata_Subclass_Tree_Thr=1000.json','r') as f:
    subclass_mapping = json.load(f)
def flatten_tree(subtree, flattened_subtree=None, level=0):
    if not flattened_subtree:
        flattened_subtree = []
    if type(subtree['subclasses']) == str:
        return flattened_subtree
    
    if subtree['entity_id'] == subtree['label']:
        label = 'NO_LABEL'
    else:
        label = subtree['label']
    
    flattened_subtree.append((subtree['entity_id'],label,level))
    
    if subtree['subclasses'] is None:
        pass
    else:
        for subclass_id, subclass_tree in subtree['subclasses'].items():
            flattened_subtree = flatten_tree(subclass_tree, flattened_subtree=flattened_subtree, level=level+1)
    return flattened_subtree

flat_subclass_mapping = {}
for k, v in subclass_mapping.items():
    flat_subclass_mapping[k] = flatten_tree(v)
    print(
        'Parsing subclass tree of',
        v['label'],
        '('+v['entity_id']+')',
        'and found',
        len(flat_subclass_mapping[k]),
        'subclasses.'
    )

flat_subclass_mapping.keys()
import numpy as np

subclass_mapping_overlap = np.empty(shape=[len(flat_subclass_mapping),len(flat_subclass_mapping)])
overlaps = {}
for i, k1 in enumerate(flat_subclass_mapping.keys()):
    overlaps[k1] = {}
    for j, k2 in enumerate(flat_subclass_mapping.keys()):
        overlaps[k1][k2] = []
        total_size = len(flat_subclass_mapping[k1])
        overlap_size = 0
        k2_entity_ids = [e[0] for e in flat_subclass_mapping[k2]]
        for e in flat_subclass_mapping[k1]:
            if e[0] in k2_entity_ids:
                overlaps[k1][k2].append(e)
                overlap_size+=1
        subclass_mapping_overlap[i,j] = overlap_size/total_size
import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(10,5))
labels = [(v[0][1]+'('+v[0][0]+')') for k,v in flat_subclass_mapping.items()]

g = sns.heatmap(
    subclass_mapping_overlap,
    xticklabels = labels,
    yticklabels = labels,
    ax=ax
)

flat_subclass_mapping.keys()

def get_subclass_root(entity_id, flat_subclass_mapping=flat_subclass_mapping, at_level=None):
    try:
        #pdb.set_trace()
        overlapping_cases = {'Q41176':False,'Q1248784':False,'Q2095':False,'Q16521':False}
        for k in flat_subclass_mapping.keys():
            if at_level:
                subclasses_entity_ids = [e[0] for e in flat_subclass_mapping[k] if e[2] == at_level[k]]
            else:
                subclasses_entity_ids = [e[0] for e in flat_subclass_mapping[k]]
            if entity_id in subclasses_entity_ids:
                if k not in overlapping_cases.keys():
                    return k
                else:
                    overlapping_cases[k] = True

        if overlapping_cases['Q41176'] or overlapping_cases['Q1248784']:
            if overlapping_cases['Q1248784']: #Airport
                return 'Q1248784' #Airport
            else:
                return 'Q41176'#Building

        if overlapping_cases['Q2095'] or overlapping_cases['Q16521']:
            if overlapping_cases['Q16521']: #Taxon
                return 'Q16521' #Taxon
            else:
                return 'Q2095'#Food

        return 'NONE'
    except ValueError as e:
        return 'ERROR: '+str(e)

from collections import Counter
for k in flat_subclass_mapping.keys():
    print(flat_subclass_mapping[k][0])
    print(Counter(map(
        lambda x: get_subclass_root(x, flat_subclass_mapping),
        [e[0] for e in flat_subclass_mapping[k]]
    )))
    print()

# Import
from pandarallel import pandarallel

# Initialization
pandarallel.initialize(progress_bar=True)

P31_df['root_class'] = P31_df.class_entity.parallel_apply(lambda x : get_subclass_root(x, flat_subclass_mapping))
P31_df.to_csv('P31_df.csv', index=None)

import WDV.WikidataClaims.wikidata_utils as wdutils
import seaborn as sns
wdAPI = wdutils.CachedWikidataAPI()
import matplotlib.pyplot as plt

root_class_counter = {}
for k in flat_subclass_mapping.keys():
    P31_df_k = P31_df[P31_df.root_class == k]
    root_class_counter[k] =  len(P31_df_k.entity_id.unique())   

subclass_instance_rate = []
for k,v in root_class_counter.items():
    if k == 'NONE':
        continue
    print(
        flat_subclass_mapping[k][0][1],
        'has', len(flat_subclass_mapping[k]),'subclasses in its tree and',
        root_class_counter[k], 'entities in total. This makes',
        round(root_class_counter[k]/len(flat_subclass_mapping[k]),2),'instances per subclass.\n'
    )
    subclass_instance_rate.append({
        'name': flat_subclass_mapping[k][0][1] + '(' + k + ')',
        'n_entities': root_class_counter[k],
        'n_subclasses': len(flat_subclass_mapping[k]),
        'subclass_entity_ratio': round(root_class_counter[k]/len(flat_subclass_mapping[k]),2)
    })

fig, ax = plt.subplots(1,3,figsize=(20,4))

subclass_instance_rate_df = pd.DataFrame(subclass_instance_rate)
g = sns.barplot(
    data=subclass_instance_rate_df.sort_values('n_entities'),
    x='name',
    y ='n_entities',
    ax=ax[0]
)
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90, ha='right')

g = sns.barplot(
    data=subclass_instance_rate_df.sort_values('n_subclasses'),
    x='name',
    y ='n_subclasses',
    ax=ax[1]
)

ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90, ha='right')

g = sns.barplot(
    data=subclass_instance_rate_df.sort_values('subclass_entity_ratio'),
    x='name',
    y ='subclass_entity_ratio',
    ax=ax[2]
)

ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=90, ha='right')

plt.show()

max_levels = {k:max([i[2] for i in v]) for k,v in flat_subclass_mapping.items()}
max_levels

cursor.execute(
    '''
    select *
    from claims
    where property_id="P106" and claim_rank!="deprecated"
    '''
)
P106_df = pd.DataFrame(cursor.fetchall())
P106_df.columns = claims_columns

P106_df['object_entity'] = P106_df.apply(
        lambda x : convert_datavalue(x['datavalue'], x['datatype']) , axis=1
)

profession_themes = ['Q11631', 'Q2066131', 'Q483501', 'Q82955']

# Total in our sample:
in_profession_entities = {}
for profession in profession_themes:
    in_profession_entities[profession] = set()
    #print('Gathering',flat_subclass_mapping[profession][0][1],'profession.')
    for subclass in flat_subclass_mapping[profession]:
        subclass_id = subclass[0]
        in_profession_entities[profession].update(P106_df[P106_df.object_entity == subclass_id].entity_id)
    print(
        'Theme',
        flat_subclass_mapping[profession][0][1],
        'has number of entities:',
        len(in_profession_entities[profession])
    )

import pickle
with open('in_profession_entities.p','wb') as f:
    pickle.dump(in_profession_entities, f)

import random
random.seed(42)
for profession in profession_themes:
    print('Samples from',flat_subclass_mapping[profession][0][1],'using instance-of route:')
    sample_from = P31_df[P31_df.root_class == profession].entity_id
    print(sample_from.sample(min(10,sample_from.shape[0])).values)

final_entity_list = {}
# NON-FOOD AND NON-PROFESSION THEMES
P31_themes = ['Q1248784', 'Q41176', 'Q515', 'Q1114461', 'Q4989906', 'Q12973014',
              'Q3918', 'Q47461344', 'Q6999', 'Q334166', 'Q16521',
              'Q79007', 'Q3305213', 'Q11173', 'Q8502']

for theme in P31_themes:
    print('Loading',flat_subclass_mapping[theme][0][1],'theme.')
    final_entity_list[theme] = set()
    final_entity_list[theme].update(P31_df[P31_df.root_class == theme].entity_id.values)

# PROFESSION THEMES
for theme in profession_themes:
    final_entity_list[theme] = in_profession_entities[theme]

# FOOD THEME
final_entity_list['Q2095'] = set()
final_entity_list['Q2095'].update([e[0] for e in flat_subclass_mapping['Q2095']])

for theme in flat_subclass_mapping:
    print('Theme',flat_subclass_mapping[theme][0][1], 'has', len(final_entity_list[theme]), 'entities.')
    
with open('final_entity_list.p','wb') as f:
    pickle.dump(final_entity_list, f)

### 3. Creating_wikidat_claim_datasets
import pandas as pd
import sqlite3
import json
import pickle

import ast
import WDV.WikidataClaims.wikidata_utils as wdutils
from importlib import reload  
reload(wdutils)
DB_PATH = '../wikidata_claims_refs_parsed.db'
claims_columns = ['entity_id','claim_id','rank','property_id','datatype','datavalue']
Wd_API = wdutils.CachedWikidataAPI(cache_path='../wikidata_entity_cache.p',save_every_x_queries=1000000)
Wd_API.languages = ['en']
with open('data/final_entity_list.p','rb') as f:
    final_entity_list = pickle.load(f)
root_entity_labels = {
    'Q1248784':'Airport',
    'Q11631':'Astronaut',
    'Q41176':'Building',
    'Q515':'City',
    'Q1114461':'ComicsCharacter',
    'Q2095':'Food',
    'Q4989906':'Monument',
    'Q12973014':'SportsTeam',
    'Q3918':'University',
    'Q47461344':'WrittenWork',
    'Q2066131':'Athlete',
    'Q483501':'Artist',
    'Q6999':'CelestialBody',
    'Q334166':'MeanOfTransportation',
    'Q82955':'Politician',
    'Q16521':'Taxon',
    'Q79007':'Street',
    'Q3305213':'Painting',
    'Q11173':'ChemicalCompound',
    'Q8502':'Mountain'
}
properties_to_remove = {
    'general':[
        'P31', # - instance of
        'P279',# - subclass of
        'P373',# - commons category
        'P910',# - Topic's main category
        'P7561',# - category for the interior of the item
        'P5008',# - on focus list of Wikimedia project
        'P2670',# -  has parts of the class
        'P1740',# -  category for films shot at this location
        'P1612',# -  Commons Institution page
        'P8989',# -  category for the view of the item
        'P2959',# -  permanent duplicated item
        'P7867',# -  category for maps
        'P935' ,# -  Commons gallery
        'P1472',#  -  Commons Creator page
        'P8596',# category for the exterior of the item
        'P5105',# Deutsche Bahn station category
        'P8933',# category for the view from the item
        'P642',# of
        'P3876',# category for alumni of educational institution
        'P1791',# category of people buried here
        'P7084',# related category
        'P1465',# category for people who died here
        'P1687',# Wikidata property
        'P6104',# maintained by WikiProject
        'P4195',# category for employees of the organization
        'P1792',# category of associated people
        'P5869',# model item
        'P1659',# see also
        'P1464',# category for people born here
        'P2354',# has list
        'P1424',# topic's main template
        'P7782',# category for ship name
        'P179',# part of the series
        'P7888',# merged into
        'P6365',# member category
        'P8464',# content partnership category
        'P360',# is a list of
        'P805',# statement is subject of
        'P8703',# entry in abbreviations table
        'P1456',# list of monuments
        'P1012',# including
        'P1151',# topic's main Wikimedia portal
        'P2490',# page at OSTIS Belarus Wiki
        'P593',# HomoloGene ID
        'P8744',# economy of topic
        'P2614',# World Heritage criteria
        'P2184',# history of topic
        'P9241',# demographics of topic
        'P487',#Unicode character
        'P1754',#category related to list
        'P2559',#Wikidata usage instructions
        'P2517',#category for recipients of this award
        'P971',#category combines topics
        'P6112',# category for members of a team
        'P4224',#category contains
        'P301',#category's main topic
        'P1753',#list related to category
        'P1423',#template has topic
        'P1204',#Wikimedia portal's main topic
        'P3921',#Wikidata SPARQL query equivalent
        'P1963',#properties for this type
        'P5125',#Wikimedia outline
        'P3176',#uses property
        'P8952',#inappropriate property for this type
        'P2306',#property
        'P5193',#Wikidata property example for forms
        'P5977',#Wikidata property example for senses
    ],
    'specific': {}
}

for theme in root_entity_labels.keys():
    properties_to_remove['specific'][theme] = []
    

# Specific predicate for AUSTRONAUT
properties_to_remove['specific']['Q11631'] = [
    'P598',#commander of (DEPRECATED)
]

if False:
    db = sqlite3.connect(DB_PATH)
    cursor = db.cursor()
    # To see how many out of the total number of stored claims we are excluding by removing the general properties
    sql_query = "select count(*) from claims where property_id in $1;"
    sql_query = sql_query.replace('$1', '(' + ','.join([('"' + e + '"') for e in properties_to_remove['general']]) + ')')
    cursor.execute(sql_query)
    print(cursor.fetchall()) #22343280
    # The total is 256557241
    db.close()
    
print('Removing the',len(properties_to_remove['general']),'properties deemed as ontological or unverbalisable',
     'WOULD leave us with',round((256557241-22343280)/256557241*100, 2),'% of the original claims accross all Wikidata.')
print('However the removal is not done here and not for all Wikidata. It is best to measure the percentage of claim',
    'coverage after all removals took place, per theme, which is the next cell')
db = sqlite3.connect(DB_PATH)
cursor = db.cursor()

theme_dfs = {}

try:
    for theme, theme_label in list(root_entity_labels.items())[:]:

        print('Processing',theme_label)

        sql_query = "select * from claims where entity_id in $1;"
        sql_query = sql_query.replace('$1', '(' + ','.join([('"' + e + '"') for e in final_entity_list[theme]]) + ')')


        cursor.execute(sql_query)
        theme_df = pd.DataFrame(cursor.fetchall())
        theme_df.columns = claims_columns
        
        original_theme_df_size = theme_df.shape[0]
        last_stage_theme_df_size = original_theme_df_size

        print('-    Removing deprecated')

        # Remove deprecated
        theme_df = theme_df[theme_df['rank'] != 'deprecated'].reset_index(drop=True)
        print(
            '    -    Percentage of deprecated:',
            round((last_stage_theme_df_size-theme_df.shape[0])/original_theme_df_size*100, 2), '%'
        )
        last_stage_theme_df_size = theme_df.shape[0]

        print('-    Removing bad datatypes')

        # Remove external_ids, commonsMedia (e.g. photos), globe-coordinates, urls
        bad_datatypes = ['commonsMedia','external-id','globe-coordinate','url', 'wikibase-form',
                         'geo-shape', 'math', 'musical-notation', 'tabular-data', 'wikibase-sense']
        theme_df = theme_df[
            theme_df['datatype'].apply(
                lambda x : x not in bad_datatypes
            )
        ].reset_index(drop=True)
        print(
            '    -    Percentage of bad datatypes:',
            round((last_stage_theme_df_size-theme_df.shape[0])/original_theme_df_size*100, 2), '%'
        )
        last_stage_theme_df_size = theme_df.shape[0]

        print('-    Removing bad properties')

        # Remove specific properties such as P31 and P279
        theme_df = theme_df[
            theme_df['property_id'].apply(
                lambda x : (x not in properties_to_remove['general']) and (x not in properties_to_remove['specific'][theme])
            )
        ].reset_index(drop=True)
        print(
            '    -    Percentage of ontology (non-domain) properties:',
            round((last_stage_theme_df_size-theme_df.shape[0])/original_theme_df_size*100, 2), '%'
        )
        last_stage_theme_df_size = theme_df.shape[0]
        
        print('-    Removing somevalue/novalue')

        # Remove novalue and somevalue
        theme_df = theme_df[
            theme_df['datavalue'].apply(
                lambda x : x not in ['somevalue', 'novalue']
            )
        ].reset_index(drop=True)
        print(
            '    -    Percentage of somevalue/novalue:',
            round((last_stage_theme_df_size-theme_df.shape[0])/original_theme_df_size*100, 2), '%'
        )
        last_stage_theme_df_size = theme_df.shape[0]
        
        print(
            'After all removals, we keep',
            round(last_stage_theme_df_size/original_theme_df_size*100, 2),
            'percent of total claims for theme', theme, '-', theme_label, '\n'
        )
        
        theme_dfs[theme] = theme_df
except Exception as e:
    raise e
finally:
    db.close()
predicate_theme_dfs = {}

for theme, theme_df in theme_dfs.items():
    
    predicate_theme_df = theme_df.groupby('property_id').count()[['entity_id']]\
        .sort_values('entity_id', ascending=False).reset_index()
    
    predicate_theme_df.columns = ['property_id','frequency_count']
    predicate_theme_df['property_label'] = predicate_theme_df['property_id'].apply(lambda x: Wd_API.get_label(x)[0])
    predicate_theme_df['frequency_percentage'] = predicate_theme_df['frequency_count'].\
        apply(lambda x: x/theme_df.shape[0]*100)

    predicate_theme_dfs[theme] = predicate_theme_df
    for theme in theme_dfs.keys():
        theme_dfs[theme].to_csv('./data/theme_dfs/'+ theme + '_claim_theme_df.csv', index=False)
        predicate_theme_dfs[theme].to_csv('./data/theme_dfs/'+ theme + '_predicate_theme_df.csv', index=False)
        print('Entity',theme,'-',root_entity_labels[theme],'has',
            theme_dfs[theme].shape[0],'claims and',predicate_theme_dfs[theme].shape[0],'predicates.')
theme_dfs, predicate_theme_dfs = {}, {}
for theme in root_entity_labels.keys():
    theme_dfs[theme] = pd.read_csv('./data/theme_dfs/'+ theme + '_claim_theme_df.csv')
    predicate_theme_dfs[theme] = pd.read_csv('./data/theme_dfs/'+ theme + '_predicate_theme_df.csv')
    print('Entity',theme,'-',root_entity_labels[theme],'has',
          theme_dfs[theme].shape[0],'claims and',predicate_theme_dfs[theme].shape[0],'predicates.')
import math
 
# SUPPORTED CONFIDENCE LEVELS: 50%, 68%, 90%, 95%, and 99%
confidence_level_constant = [50,.67], [68,.99], [90,1.64], [95,1.96], [99,2.57]
 
# CALCULATE THE SAMPLE SIZE
def sample_size(population_size, confidence_level, confidence_interval):
    Z = 0.0
    p = 0.5
    e = confidence_interval/100.0
    N = population_size
    n_0 = 0.0
    n = 0.0
 
    # LOOP THROUGH SUPPORTED CONFIDENCE LEVELS AND FIND THE NUM STD
    # DEVIATIONS FOR THAT CONFIDENCE LEVEL
    for i in confidence_level_constant:
        if i[0] == confidence_level:
            Z = i[1]

    if Z == 0.0:
        return -1

    # CALC SAMPLE SIZE
    n_0 = ((Z**2) * p * (1-p)) / (e**2)

    # ADJUST SAMPLE SIZE FOR FINITE POPULATION
    n = n_0 / (1 + ((n_0 - 1) / float(N)) )

    return int(math.ceil(n)) # THE SAMPLE SIZE

predicate_theme_df[predicate_theme_df['frequency_percentage'] >= THR].property_id.values
import math
THR = 0.05
SEED = 42
sampled_dfs = {}

for theme in list(root_entity_labels.keys())[:]:
    print('Processing theme',theme,'-',root_entity_labels[theme])
    # First get list of predicates showing up only over THR %
    predicate_theme_df = predicate_theme_dfs[theme]
    all_predicates = predicate_theme_df.property_id.values
    usual_predicates = predicate_theme_df[predicate_theme_df['frequency_percentage'] >= THR].property_id.values
    # Get claim dataset with only these non-rare predicates
    theme_df = theme_dfs[theme]
    usual_theme_df = theme_df[theme_df.apply(lambda x: x['property_id'] in usual_predicates, axis=1)]
    # Check what percentage of original claims remains.
    print(
        'After filtering, there remain',
        round(usual_theme_df.shape[0] / theme_df.shape[0], 4),
        '% of original claims.'
    )
    assert usual_theme_df.shape[0] / theme_df.shape[0] > 0.98
    # Also check how many individual predicates we would have,
    # as we would need at least a few examples of each (or to increase THR)
    print('After filtering, we are looking at', len(usual_predicates), 'filtered predicates out of originally', len(all_predicates))
    # Print the total amount of claims left and the sample size according to cohram's.
    print('After filtering, we estimate there are', usual_theme_df.shape[0]*5, 'filtered claims on the whole Wikidata.')
    ss = sample_size(usual_theme_df.shape[0], 95, 5)
    print('A good sample size (95% ci, 5% me) would be', ss, 'samples.')
    # Check how many samples of each predicate we would have, hoping it's at least 3.
    ss_per_predicate = math.floor(ss/len(usual_predicates))
    print('That is at least', ss_per_predicate, 'samples per filtered predicate.')
    assert ss_per_predicate > 3
    
    print('Generating filtered and sampled claim dataframe.')
    sampled_dfs[theme] = []
    remaining_ss = ss
    for pred in usual_predicates[::-1]:
        usual_theme_df_this_pred = usual_theme_df[usual_theme_df['property_id'] == pred]
        actual_ss_this_pred = min(ss_per_predicate, usual_theme_df_this_pred.shape[0])
        #sampled_df = get_sampled_df(
        #    df = usual_theme_df_this_pred,
        #    sample_size = actual_ss_this_pred,
        #    random_state = SEED
        #)
        sampled_df = usual_theme_df_this_pred.sample(actual_ss_this_pred, random_state = SEED)
        sampled_df['sampling_weight_vb'] = '|'.join([
            str(usual_theme_df_this_pred.shape[0]),
            str(actual_ss_this_pred)
        ])
        sampled_df['sampling_weight'] = usual_theme_df_this_pred.shape[0]/actual_ss_this_pred
        sampled_dfs[theme].append(sampled_df)
        
        remaining_ss = remaining_ss - actual_ss_this_pred
        #print(remaining_ss, len(sampled_dfs[theme]), actual_ss_this_pred)
        if remaining_ss > 0:
            ss_per_predicate = math.ceil(remaining_ss/(len(usual_predicates) - len(sampled_dfs[theme])))
        
    sampled_dfs[theme] = pd.concat(sampled_dfs[theme])
    
    print('-'*20)

    print('Saving',len(sampled_dfs),'sampled Dataframes.')
for theme, sample_df in sampled_dfs.items():
    print('Size of sample dataframe of theme',theme,'-',root_entity_labels[theme], ':', str(sample_df.shape))
    sample_df.to_csv('./data/sampled_dfs/'+ theme + '_sampled_df.csv', index=False)

sampled_dfs = {}
print('Loading sampled Dataframes.')
for theme in root_entity_labels.keys():
    sampled_dfs[theme] = pd.read_csv('./data/sampled_dfs/'+ theme + '_sampled_df.csv')
    print('Size of sample dataframe of theme',theme,'-',root_entity_labels[theme], ':', str(sampled_dfs[theme].shape))

Wd_API.x_queries_passed = Wd_API.save_every_x_queries
Wd_API.save_entity_cache()

def turn_to_century_or_millennium(y, mode):
    y = str(y)
    if mode == 'C':
        div = 100
        group = int(y.rjust(3, '0')[:-2])
        mode_name = 'century'
    elif mode == 'M':
        div = 1000
        group = int(y.rjust(4, '0')[:-3])
        mode_name = 'millenium'
    else:        
        raise ValueError('Use mode = C for century and M for millennium')
        
    if int(y)%div != 0:
        group += 1
    group = str(group)

    group_suffix = (
        'st' if group[-1] == '1' else (
            'nd' if group[-1] == '2' else (
                'rd' if group[-1] == '3' else 'th'
            )
        )
    )

    return ' '.join([group+group_suffix, mode_name])

import ast
import pdb
from datetime import datetime
def get_object_label_given_datatype(row):
    dt = row['datatype']
    dv = row['datavalue']
    
    dt_types = ['wikibase-item', 'monolingualtext', 'quantity', 'time', 'string']
    if dt not in dt_types:
        print(dt)
        raise ValueError
    else:
        try:
            if dt == dt_types[0]:
                return Wd_API.get_label(ast.literal_eval(dv)['value']['id'], True) #get label here
            elif dt == dt_types[1]:
                dv = ast.literal_eval(dv)
                return (dv['value']['text'], dv['value']['language'])
            elif dt == dt_types[2]:
                dv = ast.literal_eval(dv)
                amount, unit = dv['value']['amount'], dv['value']['unit']
                if amount[0] == '+':
                    amount = amount[1:]
                if str(unit) == '1':
                    return (str(amount), 'en')
                else:
                    unit_entity_id = unit.split('/')[-1]
                    unit = Wd_API.get_label(unit_entity_id, True)#get label here
                    return (' '.join([amount, unit[0]]), unit[1])
            elif dt == dt_types[3]:
                dv = ast.literal_eval(dv)
                time = dv['value']['time']
                timezone = dv['value']['timezone']
                precision = dv['value']['precision']
                assert dv['value']['after'] == 0 and dv['value']['before'] == 0

                sufix = 'BC' if time[0] == '-' else ''
                time = time[1:]

                if precision == 11: #date
                    return (datetime.strptime(time, '%Y-%m-%dT00:00:%SZ').strftime('%d/%m/%Y') + sufix, 'en')
                elif precision == 10: #month
                    try:
                        return (datetime.strptime(time, '%Y-%m-00T00:00:%SZ').strftime("%B of %Y") + sufix, 'en')
                    except ValueError:
                        return (datetime.strptime(time, '%Y-%m-%dT00:00:%SZ').strftime("%B of %Y") + sufix, 'en')
                elif precision == 9: #year
                    try:
                        return (datetime.strptime(time, '%Y-00-00T00:00:%SZ').strftime('%Y') + sufix, 'en')
                    except ValueError:
                        return (datetime.strptime(time, '%Y-%m-%dT00:00:%SZ').strftime('%Y') + sufix, 'en')
                elif precision == 8: #decade
                    try:
                        return (datetime.strptime(time, '%Y-00-00T00:00:%SZ').strftime('%Y')[:-1] +'0s' + sufix, 'en')
                    except ValueError:
                        return (datetime.strptime(time, '%Y-%m-%dT00:00:%SZ').strftime('%Y')[:-1] +'0s' + sufix, 'en')
                elif precision == 7: #century
                    try:
                        parsed_time = datetime.strptime(time, '%Y-00-00T00:00:%SZ')
                    except ValueError:
                        parsed_time = datetime.strptime(time, '%Y-%m-%dT00:00:%SZ')
                    finally:                        
                        return (turn_to_century_or_millennium(
                            parsed_time.strftime('%Y'), mode='C'
                        ) + sufix, 'en')
                elif precision == 6: #millennium
                    try:
                        parsed_time = datetime.strptime(time, '%Y-00-00T00:00:%SZ')
                    except ValueError:
                        parsed_time = datetime.strptime(time, '%Y-%m-%dT00:00:%SZ')
                    finally:                        
                        return (turn_to_century_or_millennium(
                            parsed_time.strftime('%Y'), mode='M'
                        ) + sufix, 'en')
                elif precision == 4: #hundred thousand years 
                    timeint = int(datetime.strptime(time, '%Y-00-00T00:00:%SZ').strftime('%Y'))
                    timeint = round(timeint/1e5,1)
                    return (str(timeint) + 'hundred thousand years' + sufix, 'en')
                elif precision == 3: #million years 
                    timeint = int(datetime.strptime(time, '%Y-00-00T00:00:%SZ').strftime('%Y'))
                    timeint = round(timeint/1e6,1)
                    return (str(timeint) + 'million years' + sufix, 'en')
                elif precision == 0: #billion years 
                    timeint = int(datetime.strptime(time, '%Y-00-00T00:00:%SZ').strftime('%Y'))
                    timeint = round(timeint/1e9,1)
                    return (str(timeint) + 'billion years' +sufix, 'en')
            elif dt == dt_types[4]:
                return (ast.literal_eval(dv)['value'], 'en')
        except ValueError as e:
            #pdb.set_trace()
            raise e
            
def get_object_desc_given_datatype(row):
    dt = row['datatype']
    dv = row['datavalue']
    
    dt_types = ['wikibase-item', 'monolingualtext', 'quantity', 'time', 'string']
    if dt not in dt_types:
        print(dt)
        raise ValueError
    else:
        try:
            if dt == dt_types[0]:
                return Wd_API.get_desc(ast.literal_eval(dv)['value']['id']) #get label here
            elif dt == dt_types[1]:
                return ('no-desc', 'none')
            elif dt == dt_types[2]:
                dv = ast.literal_eval(dv)
                amount, unit = dv['value']['amount'], dv['value']['unit']
                if amount[0] == '+':
                    amount = amount[1:]
                if str(unit) == '1':
                    return ('no-desc', 'none')
                else:
                    unit_entity_id = unit.split('/')[-1]
                    return Wd_API.get_desc(unit_entity_id)
            elif dt == dt_types[3]:
                return ('no-desc', 'none')
            elif dt == dt_types[4]:
                return ('no-desc', 'none')
        except ValueError as e:
            #pdb.set_trace()
            raise e
            
def get_object_alias_given_datatype(row):
    dt = row['datatype']
    dv = row['datavalue']
    
    dt_types = ['wikibase-item', 'monolingualtext', 'quantity', 'time', 'string']
    if dt not in dt_types:
        print(dt)
        raise ValueError
    else:
        try:
            if dt == dt_types[0]:
                return Wd_API.get_alias(ast.literal_eval(dv)['value']['id']) #get label here
            elif dt == dt_types[1]:
                return ('no-alias', 'none')
            elif dt == dt_types[2]:
                dv = ast.literal_eval(dv)
                amount, unit = dv['value']['amount'], dv['value']['unit']
                if amount[0] == '+':
                    amount = amount[1:]
                if str(unit) == '1':
                    return ('no-alias', 'none')
                else:
                    unit_entity_id = unit.split('/')[-1]
                    return Wd_API.get_alias(unit_entity_id)
            elif dt == dt_types[3]:
                dv = ast.literal_eval(dv)
                time = dv['value']['time']
                timezone = dv['value']['timezone']
                precision = dv['value']['precision']
                assert dv['value']['after'] == 0 and dv['value']['before'] == 0

                sufix = 'BC' if time[0] == '-' else ''
                time = time[1:]

                if precision == 11: #date
                    return ([
                        datetime.strptime(time, '%Y-%m-%dT00:00:%SZ').strftime('%-d of %B, %Y') + sufix,
                        datetime.strptime(time, '%Y-%m-%dT00:00:%SZ').strftime('%d/%m/%Y (dd/mm/yyyy)') + sufix,
                        datetime.strptime(time, '%Y-%m-%dT00:00:%SZ').strftime('%b %-d, %Y') + sufix
                    ], 'en')
                else: #month
                    return ('no-alias', 'none')
            elif dt == dt_types[4]:
                return ('no-alias', 'none')
        except ValueError as e:
            #pdb.set_trace()
            raise e
        
Wd_API.languages = ['en']
Wd_API.save_every_x_queries = 1000000000000000000000
for theme in sampled_dfs.keys():
    print('Acquiring labels, descriptions and aliases of',theme,'-',root_entity_labels[theme])

    print('    - Subject')
    sampled_dfs[theme]['entity_label'] = sampled_dfs[theme]['entity_id'].apply(lambda x: Wd_API.get_label(x, True))
    sampled_dfs[theme]['entity_desc'] = sampled_dfs[theme]['entity_id'].apply(lambda x: Wd_API.get_desc(x))
    sampled_dfs[theme]['entity_alias'] = sampled_dfs[theme]['entity_id'].apply(lambda x: Wd_API.get_alias(x))
    print('    - Predicate')
    sampled_dfs[theme]['property_label'] = sampled_dfs[theme]['property_id'].apply(lambda x: Wd_API.get_label(x, True))
    sampled_dfs[theme]['property_desc'] = sampled_dfs[theme]['property_id'].apply(lambda x: Wd_API.get_desc(x))
    sampled_dfs[theme]['property_alias'] = sampled_dfs[theme]['property_id'].apply(lambda x: Wd_API.get_alias(x))
    print('    - Object')
    sampled_dfs[theme]['object_label'] = sampled_dfs[theme].apply(get_object_label_given_datatype, axis=1)
    sampled_dfs[theme]['object_desc'] = sampled_dfs[theme].apply(get_object_desc_given_datatype, axis=1)
    sampled_dfs[theme]['object_alias'] = sampled_dfs[theme].apply(get_object_alias_given_datatype, axis=1)
    
    sampled_dfs[theme]['theme_entity_id'] = theme
    sampled_dfs[theme]['theme_entity_label'] = root_entity_labels[theme]


    no_subject_label_perc = sampled_dfs[theme][sampled_dfs[theme]['entity_label'].apply(lambda x : x[0] == 'no-label')].shape[0]/sampled_dfs[theme].shape[0]*100
    print('    - No subject label %:',no_subject_label_perc,'%')

    no_predicate_label_perc = sampled_dfs[theme][sampled_dfs[theme]['property_label'].apply(lambda x : x[0] == 'no-label')].shape[0]/sampled_dfs[theme].shape[0]*100
    print('    - No predicate label %:',no_predicate_label_perc,'%')

    no_object_label_perc = sampled_dfs[theme][sampled_dfs[theme]['object_label'].apply(lambda x : x[0] == 'no-label')].shape[0]/sampled_dfs[theme].shape[0]*100
    print('    - No object label %:',no_object_label_perc,'%')
    
    
sampled_df = pd.concat(sampled_dfs.values()).reset_index(drop=True)

sampled_df[['entity_label', 'entity_label_lan']] = pd.DataFrame(sampled_df.entity_label.tolist(), index=sampled_df.index)
sampled_df[['property_label', 'property_label_lan']] = pd.DataFrame(sampled_df.property_label.tolist(), index=sampled_df.index)
sampled_df[['object_label', 'object_label_lan']] = pd.DataFrame(sampled_df.object_label.tolist(), index=sampled_df.index)

sampled_df[['entity_desc', 'entity_desc_lan']] = pd.DataFrame(sampled_df.entity_desc.tolist(), index=sampled_df.index)
sampled_df[['property_desc', 'property_desc_lan']] = pd.DataFrame(sampled_df.property_desc.tolist(), index=sampled_df.index)
sampled_df[['object_desc', 'object_desc_lan']] = pd.DataFrame(sampled_df.object_desc.tolist(), index=sampled_df.index)

sampled_df[['entity_alias', 'entity_alias_lan']] = pd.DataFrame(sampled_df.entity_alias.tolist(), index=sampled_df.index)
sampled_df[['property_alias', 'property_alias_lan']] = pd.DataFrame(sampled_df.property_alias.tolist(), index=sampled_df.index)
sampled_df[['object_alias', 'object_alias_lan']] = pd.DataFrame(sampled_df.object_alias.tolist(), index=sampled_df.index)

sampled_df.to_csv('./data/sampled_df_pre_verbalisation.csv', index=False)