import pandas as pd
import numpy as np
from ast import literal_eval as leval
import seaborn as sns
from tqdm.auto import tqdm
tqdm.pandas()
from matplotlib import pyplot as plt
import re

def WTR_complier():
    reference_text_df = pd.read_csv('../text_extraction/reference_html_as_sentences_df.csv')
    claim_data_df = pd.read_csv('../text_extraction/text_reference_claims_df.csv')

    reference_text_df.info()
    reference_text_df_trim = reference_text_df.drop(
        ['error_msg','code','content-type','reason','language_crawl','language_crawl_score',
        'sampling_weight_vb', 'sampling_weight', 'extracted_sentences', 'extracted_text', 'nlp_sentences',
        'nlp_sentences_slide_2'], axis=1
    )
    reference_text_df_trim.info()

    # Replace by archived page if page was behing paywall when parsed
    reference_text_df_trim.loc[reference_text_df_trim.html.map(lambda x : '://archive.ph/' in x), 'final_url'] =\
        reference_text_df_trim.loc[reference_text_df_trim.html.map(lambda x : '://archive.ph/' in x)]\
            .html.map(lambda x : re.findall(r'http(?:s){0,1}://archive.ph/(?:[a-zA-Z0-9]*)', x)[0])

    claim_data_df.info()
    claim_data_df_trim = claim_data_df[
        [
            'reference_id','claim_id','rank','datatype','datavalue',
            'entity_id','property_id',
            'entity_label','property_label','object_label',
            'entity_alias','property_alias','object_alias',
            'entity_desc','property_desc','object_desc'
        ]
    ].copy()

    claim_data_df_trim['verb_mock'] = claim_data_df_trim.apply(
        lambda row: '$'.join([row['entity_label'], row['property_label'], row['object_label']]), axis=1
    )

    claim_data_df_trim.info()

    # REMOVE P1448 (OFFICIAL NAME), P1476 (TITLE), AND P1889 (DIFFERENT) AS THEY ARE REDUNDANT AND NON-INFORMATIVE
    #also look at the dataset creation for other properties that were deleted and delete them too

    BAD_PROPERTIES = [
        'P1448', # offical name
        'P1476', # title
        'P1889',# different
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
        'P1748',#NCI Thesaurus ID
        'P1692',#ICD-9-CM
        'P248',#stated in
    ]

    claim_data_df_trim = claim_data_df_trim.drop(473, axis='index') #No english object label or aliases
    claim_data_df_trim = claim_data_df_trim.drop(593, axis='index') #No english object label or aliases
    claim_data_df_trim = claim_data_df_trim.reset_index(drop=True)

    claim_data_df_trim_badpropdrop = claim_data_df_trim[
        ~claim_data_df_trim['property_id'].isin(BAD_PROPERTIES)
    ]
    print('Percentage [Number] of claims dropped due to bad properties')
    print(
        f'{100 - 100*claim_data_df_trim_badpropdrop.shape[0]/claim_data_df_trim.shape[0]}',
        f'[{claim_data_df_trim.shape[0] - claim_data_df_trim_badpropdrop.shape[0]}]'
    )

    # Get URLs from the references df
    wtr = pd.merge(
        reference_text_df_trim,
        claim_data_df_trim_badpropdrop,
        on='reference_id'
    )
    wtr
    wtr.reference_id.unique().shape
    # Remove duplicates of reference and verbalisation, as duplicates arise from qualifier dependancy
    wtr_duplidrop = \
        wtr.drop_duplicates(
        ['verb_mock','final_url'], keep='first'
    )

    wtr_duplidrop = wtr_duplidrop.drop(806, axis='index') #Verb will be equal

    print('Percentage [Number] of claims dropped due to duplicated verbalisation and url pair')
    print(
        f'{100 - 100*wtr_duplidrop.shape[0]/wtr.shape[0]}',
        f'[{wtr.shape[0] - wtr_duplidrop.shape[0]}]'
    )
    # Remove the three cases in archinform.net written in German
    wtr_duplidrop = wtr_duplidrop[
        ~wtr_duplidrop['final_url'].map(
            lambda x : 'www.archinform.net' in x and any([(y in x) for y in ['19632','11996','45859']])
        )
    ]
    wtr_duplidrop = wtr_duplidrop.reset_index(drop=True)
    wtr_duplidrop
    wtr_duplidrop.to_csv('WTR_non_filtered_non_annotated.csv', index=None)
    wtr_trim = wtr_duplidrop.sample(frac=1, random_state=42).drop_duplicates('reference_id').sort_index().reset_index(drop=True)
    wtr_trim.info()
    bad_netloc_aggs = [
        'witches.shca.ed.ac.uk', #Single infobox
        'en.isabart.org', #Single infobox
        'bechdeltest.com', #HAS API
        'npg.si.edu', #Image and single infobox
        'www.guidetopharmacology.org', #Single infobox
        'letterboxd.com', #Single infobox and has API
        'www.discogs.com', #Single infobox
        'vocab.getty.edu', #HAS JSON DUMPS
        'www.isfdb.org', #single infobox
        'www.npg.org.uk', #set of infoboxes
        'art.nationalgalleries.org', #image and single infobox
        'www.tate.org.uk', #image and single infobox
        'www.getty.edu', #HAS JSON DUMPS
        'memory-beta.fandom.com', #The portion with the information on Claims is just a long list of names and links
        'www.disease-ontology.org', #A single infobox
        'artgallery.yale.edu', #Image and a single infobox
        'www.imdb.com', #These are author pages and consist of a portrait, an infobox, and lists of movies
        'muckrack.com', #A very short infobox
        'live.dbpedia.org', #It's dbpedia, so it's mainly a huge infobox and there are dumps
        'dbpedia.org' #Same as above
    ]
    wtr_trim_good = wtr_trim[~wtr_trim.netloc_agg.isin(bad_netloc_aggs)].reset_index(drop=True)

    wtr_trim_good.info()