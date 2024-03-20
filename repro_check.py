import pandas as pd
import numpy as np

verb_new = pd.read_csv('verbalisation/verbalised_claims_df_final.csv')
sent_new = pd.read_json('sentence_retrieval/sentence_relevance_df.json')
text_new = pd.read_json('Prove/textual_entailment/textual_entailment_df.json')

verb_old = pd.read_csv('InProduction/verbalised_claims_df_final.csv')
sent_old = pd.read_json('InProduction/sentence_relevance_df.json')
text_old = pd.read_json('InProduction/textual_entailment_df.json')

#Compared correctd_verbalisation
new = verb_new['corrected_verbalisation']
old = verb_old['corrected_verbalisation']

comparison_result = [1 if new[i] == old
[i] else 0 for i in range(len(new))]
print(comparison_result)

#Compared sentence_relevance
new = sent_new['nlp_sentences_all_TOP_N']
old = sent_old['nlp_sentences_all_TOP_N']

aggrements = []

for i in range(len(new)):
    sentences_data1 = set([entry['sentence'] for entry in new[i]])
    sentences_data2 = set([entry['sentence'] for entry in old[i]])
    common_sentences = sentences_data1.intersection(sentences_data2)
    common_sentences_count = len(common_sentences)
    aggrements.append(common_sentences_count/5)


average = sum(aggrements) / len(aggrements)
print(average)


#Compared textual_entailment_df
new = text_new['claim_TE_label_weighted_sum_all_TOP_N']
old = text_old['claim_TE_label_weighted_sum_all_TOP_N']

comparison_result = [1 if new[i] == old
[i] else 0 for i in range(len(new))]
print(f'comparision of weighted_sum_all_TOP_N:{sum(comparison_result)/len(comparison_result)}')

new = text_new['claim_TE_label_malon_all_TOP_N']
old = text_old['claim_TE_label_malon_all_TOP_N']

comparison_result = [1 if new[i] == old
[i] else 0 for i in range(len(new))]
print(f'comparision of weighted_sum_all_TOP_N:{sum(comparison_result)/len(comparison_result)}')