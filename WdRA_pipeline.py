import numpy as np
import pandas as pd
#import scipy.stats as stats
import json
import seaborn as sns
import qwikidata
import random
import matplotlib.pyplot as plt
from qwikidata.json_dump import WikidataJsonDump
from qwikidata.linked_data_interface import get_entity_dict_from_api
from qwikidata.sparql import (get_subclasses_of_item,
                              return_sparql_query_results)
from collections import Counter
import ast
from qwikidata.linked_data_interface import LdiResponseNotOk
from qwikidata.entity import WikidataItem, WikidataProperty, WikidataLexeme
import sqlite3
from urllib3.exceptions import MaxRetryError
import threading
import time
import timeit
import traceback
import plotly.graph_objects as go
import tldextract
from urllib.parse import urlparse
import ssl
import languages_and_countries
from samplesize import sampleSize
import importlib

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

import logging
logging.basicConfig(
    filename='process.log',
    filemode='a+',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

wikidata_db = sqlite3.connect('/mnt/data/group3/wikidata/part/wikidata_apr.db')
#wikidata_db = sqlite3.connect('F:/wikidata_apr.db')

sql_cursor = wikidata_db.cursor()

def kill(conn):
    while True: 
        with open('iskill.txt','r') as f:
            time.sleep(1)
            if f.readline().strip() == 'yes':
                print('killed')
                conn.interrupt()
                break
                
th = threading.Thread(target=kill,args=[wikidata_db])
th.start()

update = False

color_palette_list = ['#009ACD', '#ADD8E6', '#63D1F4', '#0EBFE9',   
                      '#C1F0F6', '#0099CC']

def get_entity(item_id):
    while True:
        try:
            entity = get_entity_dict_from_api(item_id)
            return entity
        except ConnectionError:
            #traceback.print_exc()
            continue
        except MaxRetryError:
            #traceback.print_exc()
            time.sleep(1)
        except LdiResponseNotOk:
            #traceback.print_exc()
            return 'deleted'

def get_label(item):
    if type(item) == str:        
        entity = get_entity(item)
        if entity == 'deleted':
            return entity
        labels = entity['labels']
    elif type(item) == dict:
        labels = item['labels']
    languages = ['en','fr','es','pt','pt-br','it','de']
    for l in languages:
        if l in labels:
            return labels[l]['value']
    return 'no-label'

def get_datatype(item):
    try:
        if type(item) == str:
            entity = get_entity(item)
            if entity == 'deleted':
                return entity
            datatype = entity['datatype']
        elif type(item) == dict:
            datatype = item['datatype']
        return datatype
    except KeyError:
        return 'none'

def get_claim_values_of(item, property_id):
    if type(item) == str:
        entity = get_entity(item)
        if entity == 'deleted':
            return entity
        claims = entity['claims']
    elif type(item) == dict:
        claims = item['claims']
    if property_id in claims:
        instance_of_claims = claims[property_id]
        return [i['mainsnak']['datavalue']['value']['id'] for i in instance_of_claims]
    else:
        return []
    
def aggregate_other(df, by, count_name='count', other_thr=1):
    df_c = df.copy()
    df_c = df_c[[by,count_name]]
    total_count = df_c[count_name].sum()
    df_c['per'] = df_c[count_name].apply(lambda x: 100*x/total_count)

    other_df_c = df_c[df_c['per'] < other_thr].sum()
    other_df_c[by] = 'other'

    df_c = df_c.append(
        other_df_c, ignore_index=True
    ).sort_values('per',ascending=False).reset_index(drop=True)
    df_c = df_c[df_c['per'] >= other_thr]

    return df_c

sql_cursor.execute(
    ''' select * from claims limit 10 '''
)
head_10 = pd.DataFrame(sql_cursor.fetchall())
head_10.columns = ['entity_id','claim_id','claim_rank','property_id','datatype','datavalue']
head_10

sql_cursor.execute(
    ''' select * from refs limit 10 '''
)
head_10 = pd.DataFrame(sql_cursor.fetchall())
head_10.columns = ['reference_id','reference_property_id','reference_index','reference_datatype','reference_value']
head_10

sql_cursor.execute(
    ''' select * from claims_refs limit 10 '''
)
head_10 = pd.DataFrame(sql_cursor.fetchall())
head_10.columns = ['claim_id','reference_id']
head_10

if update:
    sql_cursor.execute(
        ''' select count(distinct claim_id) from claims indexed by claim_id_index
     '''
    )
    total_count_claims = sql_cursor.fetchall()[0][0]
else:
    total_count_claims = 195874387
print('There are {} total claims nodes'.format(total_count_claims))

if update:
    sql_cursor.execute(
        ''' select count(distinct claim_id) from claims_refs indexed by claim_refs_claim_index
     '''
    )
    total_count_claims_with_refs = sql_cursor.fetchall()[0][0]
else: 
    total_count_claims_with_refs = 151566485 
print('There are {} total claims nodes with references'.format(total_count_claims_with_refs))

total_count_claims_no_refs = total_count_claims - total_count_claims_with_refs


fig, ax = plt.subplots(figsize=[5,5])
percentages = [total_count_claims_no_refs, total_count_claims_with_refs]

labels = ['No references','References']

ax.pie(percentages, labels=labels,  
       colors=color_palette_list[0:2], autopct='%1.0f%%', 
       shadow=False, startangle=0,labeldistance=None)
ax.axis('equal')
ax.set_title("Percentage of claims with references")
ax.legend(frameon=False, bbox_to_anchor=(0,1))
plt.show()

if update:
    sql_cursor.execute(
        ''' 
        select count(distinct reference_id) from refs indexed by reference_id_index
     '''
    )
    total_count_refs = sql_cursor.fetchall()[0][0]
else:
    total_count_refs = 13371626
print('There are {} total references nodes'.format(total_count_refs))
print('For each unique reference node there are {} claims with references'.format(total_count_claims_with_refs/total_count_refs))

# One reference node may be connected to many statements. This aggregation is base on statements, meaning that
# if a reference node apears 100 times and has a Stated In property, this property is counted 100 times.
if update:
    sql_cursor.execute(
        ''' select reference_property_id, count(*) as c from refs group by reference_property_id order by c desc
     '''
    )
    reference_property_count = pd.DataFrame(sql_cursor.fetchall())
    reference_property_count.to_csv('data/reference_property_count.csv',index=False)
else:
    reference_property_count = pd.read_csv('data/reference_property_count.csv')
reference_property_count.columns = ['reference_property_id','count']
reference_property_count = reference_property_count[reference_property_count['reference_property_id'] != 'none'].reset_index(drop=True)
    
reference_property_count_c = aggregate_other(reference_property_count, by='reference_property_id', count_name='count', other_thr=0.5)
reference_property_count_c['label'] = reference_property_count_c['reference_property_id'].transform(lambda x : get_label(x) if x!= 'other' else x)

fig, ax = plt.subplots(figsize=[10,5])
percentages = reference_property_count_c['per']
labels = reference_property_count_c.apply(lambda x : ' - '.join([x['reference_property_id'],x['label']]), axis=1)

ax = sns.barplot(y=labels, x=percentages)
ax.set_title("Distribution of reference node properties")
ax.set(ylabel='Property (Wikidata ID and English label)', xlabel='Percentage of total properties.')

for p in ax.patches:
    width = p.get_width()
    ax.text(width + 1 if width < 15 else width - 1 ,
            p.get_y()+p.get_height()/2. + 0.2,
            '{:1.2f}'.format(width),
            ha="center")
    
plt.show()

interesting_prop_ids = ['P248','P854','P4656','P143']
if update:
    def f(x):
        if x in interesting_prop_ids:
            sql_cursor.execute('''
                select count(distinct reference_id)
                from refs
                where reference_property_id = '{}'
                and reference_value not in ('novalue','somevalue');
            '''.format(x))
            r = sql_cursor.fetchall()[0][0]
            return r/total_count_refs*100
        else:
            return None
    reference_property_count_c['coverage'] = reference_property_count_c['reference_property_id'].apply(f)
else:
    coverages = [85.955687, 59.876331, 5.002615, 5.106148]
    reference_property_count_c['coverage'] = reference_property_count_c['reference_property_id'].apply(
        lambda x : coverages[interesting_prop_ids.index(x)] if x in interesting_prop_ids else None
    )
reference_property_count_c

if update:
    sql_cursor.execute(
        ''' 
        select count(*) from refs where 
     '''
    )
    total_refs_values_count = sql_cursor.fetchall()[0][0]
else:
    total_refs_values_count = 42586861
print('There are {} total reference property values'.format(total_refs_values_count))

if update:
    sql_cursor.execute(
        ''' 
        select count(*) from refs where reference_value == 'somevalue'
     '''
    )
    total_refs_somevalues_count = sql_cursor.fetchall()[0][0]
else:
    total_refs_somevalues_count = 24
print('There are {} total reference property values which are somevalue'.format(total_refs_somevalues_count))

if update:
    sql_cursor.execute(
        ''' 
        select count(*) from refs where reference_value == 'novalue'
     '''
    )
    total_refs_novalues_count = sql_cursor.fetchall()[0][0]
else:
    total_refs_novalues_count = 63
print('There are {} total reference property values which are novalue'.format(total_refs_novalues_count))

# Aggregate and show results 
if update:
    sql_cursor.execute(
        ''' select reference_datatype, count(*) as c
        from refs group by reference_datatype
        order by c desc
     '''
    )
    ref_datatype_count = pd.DataFrame(sql_cursor.fetchall())
    ref_datatype_count.to_csv('data/ref_datatype_count.csv',index=False)
else:
    ref_datatype_count = pd.read_csv('data/ref_datatype_count.csv', header=None)
ref_datatype_count.columns = ['reference_datatype','count']

fig, ax = plt.subplots(figsize=[10,5])
percentages = ref_datatype_count['count'].apply(lambda x : 100*x/ref_datatype_count['count'].sum())
labels = ref_datatype_count['reference_datatype']

ax = sns.barplot(y=labels, x=percentages)
ax.set_title("Distribution of reference property value datatypes")
ax.set(ylabel='Datatype ', xlabel='Percentage of total reference property values')

for p in ax.patches:
    width = p.get_width()
    ax.text(width + 1 if width < 15 else width - 1 ,
            p.get_y()+p.get_height()/2. + 0.2,
            '{:1.2f}'.format(width),
            ha="center")
    
plt.show()

if update:
    sql_cursor.execute(
        ''' select reference_value
        from refs where reference_property_id = 'P248' 
        and reference_value not in ['novalue','somevalue']
     '''
    )
    stated_in_ids = pd.DataFrame(sql_cursor.fetchall())
    stated_in_ids[0] = stated_in_ids[0].transform(lambda x : json.loads(x.replace("'",'"'))['value']['id'])
    stated_in_ids.columns = ['stated_in_id']
    
    stated_in_ids['count'] = 1
    stated_in_ids = stated_in_ids.groupby('stated_in_id').sum().sort_values('count',ascending=False).reset_index()
    stated_in_ids['label'] = 'none'
    stated_in_ids['instance_of'] = 'none'
    stated_in_ids['subclass_of'] = 'none'
    total = stated_in_ids.shape[0]
    for i in range(total):
        print('{}%'.format((i+1)/total*100) + ' '*15, end='\r')
        if 'none' in list(stated_in_ids.loc[i,['label','instance_of','subclass_of']]):
            entity = get_entity(stated_in_ids.loc[i,'stated_in_id'])
            if type(entity) == str:
                stated_in_ids.loc[i,'label'] = entity
                stated_in_ids.loc[i,'instance_of'] = ''
                stated_in_ids.loc[i,'subclass_of'] = ''
            elif type(entity) == dict:
                stated_in_ids.loc[i,'label'] = str(get_label(entity))
                stated_in_ids.loc[i,'instance_of'] = ','.join(get_claim_values_of(entity,'P31'))
                stated_in_ids.loc[i,'subclass_of'] = ','.join(get_claim_values_of(entity,'P279'))
    stated_in_ids.to_csv('data/stated_in_refs_df.csv',index=False)
else:
    stated_in_ids = pd.read_csv('data/stated_in_refs_df.csv').fillna('')
stated_in_ids

other_thr = 0.5 #as in 1%

stated_in_ids_c = stated_in_ids.copy()
stated_in_ids_c = stated_in_ids_c.drop(['subclass_of','instance_of'],axis=1)
total_count = stated_in_ids_c['count'].sum()
stated_in_ids_c['per'] = stated_in_ids_c['count'].apply(lambda x: 100*x/total_count)

other_stated_in_ids_c = stated_in_ids_c[stated_in_ids_c['per'] < other_thr].sum()
other_stated_in_ids_c['stated_in_id'] = 'other'
other_stated_in_ids_c['label'] = 'other'

stated_in_ids_c = stated_in_ids_c.append(other_stated_in_ids_c, ignore_index=True).sort_values('per',ascending=False).reset_index(drop=True)
stated_in_ids_c = stated_in_ids_c[stated_in_ids_c['per'] >= other_thr]

fig, ax = plt.subplots(figsize=[10,5])
percentages = stated_in_ids_c['per']
labels = stated_in_ids_c.apply(lambda x : ' - '.join([x['stated_in_id'],x['label']]), axis=1)

ax = sns.barplot(y=labels, x=percentages)
ax.set_title("Distribution of stated-in sources")
ax.set(ylabel='Stated-in source', xlabel='Percentage of frequency')

for p in ax.patches:
    width = p.get_width()
    ax.text(width + 3 if width < 20 else width - 3 ,
            p.get_y()+p.get_height()/2. + 0.2,
            '{:1.2f}'.format(width),
            ha="center")
    
plt.show()

if update:
    stated_in_ids_instance_of = stated_in_ids.copy()
    stated_in_ids_instance_of = stated_in_ids_instance_of.drop(['label','subclass_of'],axis=1)
    stated_in_ids_instance_of = (stated_in_ids_instance_of.set_index(['stated_in_id', 'count'])
       .apply(lambda x: x.str.split(',').explode())
       .reset_index())

    stated_in_ids_instance_of.drop('stated_in_id',axis=1,inplace=True)
    stated_in_ids_instance_of = stated_in_ids_instance_of[['instance_of','count']]
    stated_in_ids_instance_of = (stated_in_ids_instance_of.groupby('instance_of')
                                 .sum().sort_values('count',ascending=False).reset_index())
    stated_in_ids_instance_of['label'] = 'none'
    stated_in_ids_instance_of['instance_of_of'] = 'none'
    stated_in_ids_instance_of['subclass_of_of'] = 'none'

    total = stated_in_ids_instance_of.shape[0]
    for i in range(total):
        try:
            print('{}%'.format((i+1)/total*100) + ' '*15, end='\r')
            if 'none' == stated_in_ids_instance_of.loc[i,'label']:
                if stated_in_ids_instance_of.loc[i,'instance_of'] != '':
                    entity = get_entity(stated_in_ids_instance_of.loc[i,'instance_of'])
                    stated_in_ids_instance_of.loc[i,'label'] = str(get_label(entity))
                    stated_in_ids_instance_of.loc[i,'instance_of_of'] = ','.join(get_claim_values_of(entity,'P31'))
                    stated_in_ids_instance_of.loc[i,'subclass_of_of'] = ','.join(get_claim_values_of(entity,'P279'))
        except Exception as e:
            print(e,i)
            traceback.print_exc()
            raise
    stated_in_ids_instance_of.to_csv('data/stated_in_ids_instance_of.csv', index=False)
else:
    stated_in_ids_instance_of = pd.read_csv('data/stated_in_ids_instance_of.csv').fillna('')
stated_in_ids_instance_of

other_thr = 0.25 #as in 0.25%

stated_in_ids_instance_of_c = stated_in_ids_instance_of.copy()
stated_in_ids_instance_of_c = stated_in_ids_instance_of_c.drop(['instance_of_of','subclass_of_of'],axis=1)
total_count = stated_in_ids_instance_of_c['count'].sum()
stated_in_ids_instance_of_c['per'] = stated_in_ids_instance_of_c['count'].apply(lambda x: 100*x/total_count)

other_stated_in_ids_instance_of_c = stated_in_ids_instance_of_c[stated_in_ids_instance_of_c['per'] < other_thr].sum()
other_stated_in_ids_instance_of_c['instance_of'] = 'other'
other_stated_in_ids_instance_of_c['label'] = 'other'

stated_in_ids_instance_of_c = stated_in_ids_instance_of_c.append(
    other_stated_in_ids_instance_of_c, ignore_index=True
).sort_values('per',ascending=False).reset_index(drop=True)
stated_in_ids_instance_of_c = stated_in_ids_instance_of_c[stated_in_ids_instance_of_c['per'] >= other_thr]

fig, ax = plt.subplots(figsize=[10,10])
percentages = stated_in_ids_instance_of_c['per']
labels = stated_in_ids_instance_of_c.apply(lambda x : ' - '.join([x['instance_of'],x['label']]), axis=1)

ax = sns.barplot(y=labels, x=percentages)
ax.set_title("Distribution of classes of stated-in reference objects")
ax.set(ylabel='Classes of stated-in reference objects', xlabel='Percentage of frequency')

for p in ax.patches:
    width = p.get_width()
    ax.text(width + 3 if width < 20 else width - 3 ,
            p.get_y()+p.get_height()/2. + 0.2,
            '{:1.2f}'.format(width),
            ha="center")
    
plt.show()

# In order of largest representation:
stated_in_examples_class, stated_in_examples_labels = [], []
for row in stated_in_ids_instance_of_c.itertuples():
    stated_in_examples_class.append(row.label + '({})'.format(row.instance_of))

    stated_in_Q = stated_in_ids[stated_in_ids['instance_of'].apply(lambda x : row.instance_of in x)]
    examples = list(stated_in_Q.apply(lambda x: x['label'] + '({})'.format(x['stated_in_id']), axis=1))

    if len(stated_in_Q) > 15:
        examples = random.sample(examples,15)
    stated_in_examples_labels.append(','.join(examples))
    
fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Stated-in Class", "Instance Examples"],
            font=dict(size=10),
            align="left"
        ),
        cells=dict(
            values=[stated_in_examples_class, stated_in_examples_labels],
            align = "left")
    )
])

fig.update_layout(
    height=2900,
    showlegend=False,
    title_text="Examples of each stated-in class",
)

fig.show()
