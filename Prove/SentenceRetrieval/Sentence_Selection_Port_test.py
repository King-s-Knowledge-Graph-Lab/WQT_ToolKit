import torch

args = {
    'batch_size': 32,
    'bert_pretrain':'./bert_base',
    'checkpoint': './checkpoint/model.best.32.pt',
    'dropout': 0.6,
    'bert_hidden_dim': 768,
    'max_len': 384,
}

args['cuda'] = torch.cuda.is_available()

import re

def process_sent(sentence):
    sentence = re.sub("LSB.*?RSB", "", sentence)
    sentence = re.sub("LRB\s*?RRB", "", sentence)
    sentence = re.sub("(\s*?)LRB((\s*?))", "\\1(\\2", sentence)
    sentence = re.sub("(\s*?)RRB((\s*?))", "\\1)\\2", sentence)
    sentence = re.sub("--", "-", sentence)
    sentence = re.sub("``", '"', sentence)
    sentence = re.sub("''", '"', sentence)
    
    return sentence

test_claim = [
    {"id": 137334, "evidence": [["Soul_Food_-LRB-film-RRB-", 0, "Soul Food is a 1997 American comedy drama film produced by Kenneth `` Babyface '' Edmonds , Tracey Edmonds and Robert Teitel and released by Fox 2000 Pictures .", 0.999876856803894], ["Soul_Food_-LRB-film-RRB-", 7, "In 2015 , it was announced that 20th Century Fox is planning a sequel for film called More Soul Food , written by Tillman , Jr. .", 0.1306380331516266], ["Soul_Food_-LRB-TV_series-RRB-", 1, "Created by George Tillman , Jr. and developed for television by Felicia D. Henderson , Soul Food is based upon Tillman 's childhood experiences growing up in Wisconsin , and is a continuation of his 1997 film of the same name .", -0.34905487298965454], ["Soul_Food_-LRB-film-RRB-", 1, "Featuring an ensemble cast , the film stars Vanessa L. Williams , Vivica A. Fox , Nia Long , Michael Beach , Mekhi Phifer , Jeffrey D. Sams , Irma P. Hall , Gina Ravera and Brandon Hammond .", -0.38830122351646423], ["Soul_Food_-LRB-TV_series-RRB-", 0, "Soul Food is a television drama that aired Wednesday nights on Showtime from June 28 , 2000 to May 26 , 2004 .", -0.4003089368343353]], "claim": "Fox 2000 Pictures released the film Soul Food.", "label": "SUPPORTS"},
    {"id": 178901, "evidence": [["Dawood_Ibrahim", 0, "Dawood Ibrahim LRB Marathi : \u0926\u093e\u090a\u0926 \u0907\u092c\u094d\u0930\u093e\u0939\u0940\u092e \u0915\u093e\u0938\u0915\u0930 , born Dawood Ibrahim Kaskar 26 December 1955 RRB , known as Dawood Bhai or simply Bhai is a gangster and terrorist originally from Dongri in Mumbai , India .", 0.9985103011131287], ["Dawood_Ibrahim", 5, "He heads the Indian organised crime syndicate D Company founded in Mumbai .", -0.9245980978012085], ["Namibia", 4, "Its capital and largest city is Windhoek , and it is a member state of the United Nations LRB UN RRB , the Southern African Development Community LRB SADC RRB , the African Union LRB AU RRB , and the Commonwealth of Nations .", -0.9981015920639038], ["Namibia", 0, "Namibia LRB Republik Namibia Republiek van Namibi\u00eb RRB , is a country in southern Africa whose western border is the Atlantic Ocean .", -0.9981855750083923], ["Namibia", 9, "Since then the Bantu groups in total , known as the Ovambo people , have dominated the population of the country and since the late 19th century , have constituted a large majority .", -0.9991973638534546]], "claim": "Dawood Ibrahim was born in Namibia.", "label": "NOT ENOUGH INFO"},
    {"id": 91198,  "evidence": [["Colin_Kaepernick", 6, "He remained the team 's starting quarterback for the rest of the season and went on to lead the 49ers to their first Super Bowl appearance since 1994 , losing to the Baltimore Ravens .", 0.9993929862976074],["Colin_Kaepernick", 0, "Colin Rand Kaepernick LRB LSB ` k\u00e6p\u0259rn\u026ak RSB ; born November 3 , 1987 RRB is an American football quarterback who is currently a free agent .", 0.26022613048553467],["Colin_Kaepernick", 2, "Kaepernick was selected by the San Francisco 49ers in the second round of the 2011 NFL Draft .", -0.05764467641711235],["Colin_Kaepernick", 8, "In the following seasons , Kaepernick lost and won back his starting job , with the 49ers missing the playoffs for three years consecutively .", -0.07537994533777237],["Colin_Kaepernick", 7, "During the 2013 season , his first full season as a starter , Kaepernick helped the 49ers reach the NFC Championship , losing to the Seattle Seahawks .", -0.07819903641939163]], "claim": "Colin Kaepernick became a starting quarterback during the 49ers 63rd season in the National Football League.", "label": "NOT ENOUGH INFO"},
    {"id": 166626, "evidence": [["Anne_Rice", 5, "Born in New Orleans , Rice spent much of her early life there before moving to Texas , and later to San Francisco .", 0.9998263120651245], ["Anne_Rice", 0, "Anne Rice LRB born Howard Allen Frances O'Brien ; October 4 , 1941 RRB is an American author of gothic fiction , Christian literature , and erotica .", 0.05985370650887489], ["Anne_Rice", 7, "She began her professional writing career with the publication of Interview with the Vampire in 1976 , while living in California , and began writing sequels to the novel in the 1980s .", 0.01668621599674225], ["Anne_Rice", 6, "She was raised in an observant Catholic family , but became an agnostic as a young adult .", -0.8269168138504028], ["Anne_Rice", 16, "She was married to poet and painter Stan Rice for 41 years , from 1961 until his death from brain cancer in 2002 at age 60 .", -0.852485716342926]], "claim": "Anne Rice was born in New Jersey.", "label": "NOT ENOUGH INFO"},
    {"id": 111897, "evidence": [["Telemundo", 0, "Telemundo LRB LSB tele\u02c8mundo RSB RRB is an American Spanish language terrestrial television network owned by Comcast through the NBCUniversal division NBCUniversal Telemundo Enterprises .", 0.9999074935913086], ["Telemundo", 4, "The channel broadcasts programs and original content aimed at Hispanic and Latino American audiences in the United States and worldwide , consisting of telenovelas , sports , reality television , news programming , and films either imported or Spanish dubbed .", 0.5357285737991333], ["Telemundo", 1, "It is the second largest provider of Spanish content nationwide behind American competitor Univision , with programming syndicated worldwide to more than 100 countries in over 35 languages .", 0.5267376899719238], ["Telemundo", 5, "In addition , Telemundo operates NBC Universo , a separate channel directed towards young Hispanic audiences ; Telemundo Digital Media , which distributes original programming content across mass media , the Telemundo and NBC Universo websites ; Puerto Rico telestation WKAQ TV ; and international distribution arm Telemundo Internacional .", -0.22787447273731232], ["Telemundo", 9, "The majority of Telemundo 's programs are filmed at an operated studio facility in Miami , where 85 % of the network 's telenovelas were filmed during 2011 .", -0.4263255000114441]], "claim": "Telemundo is a English-language television network.", "label": "REFUTES"},
    {"id": 171897, "evidence": [["Research", 4, "To test the validity of instruments , procedures , or experiments , research may replicate elements of prior projects or the project as a whole .", 0.9998928308486938], ["Test_validity", 1, "In the fields of psychological testing and educational testing , `` validity refers to the degree to which evidence and theory support the interpretations of test scores entailed by proposed uses of tests '' .", 0.22765414416790009], ["Experiment", 3, "There also exists natural experimental studies .", 0.16992847621440887], ["Experiment", 14, "This increases the reliability of the results , often through a comparison between control measurements and the other measurements .", 0.13860490918159485], ["Experiment", 9, "Experiments can vary from personal and informal natural comparisons LRB e.g. tasting a range of chocolates to find a favorite RRB , to highly controlled LRB e.g. tests requiring complex apparatus overseen by many scientists that hope to discover information about subatomic particles RRB .", 0.10734457522630692]], "claim": "Research is incapable of testing the validity of experiments.", "label": "REFUTES"}
]

test_inputs = [[process_sent(t['claim']), process_sent(e[2])] for t in test_claim for e in t['evidence']]
len(test_inputs)

test_inputs[0]

from pytorch_pretrained_bert.tokenization import BertTokenizer as BertTokenizerRepo
from torch.autograd import Variable

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def tok2int_sent(sentence, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""
    sent_a, sent_b = sentence
    tokens_a = tokenizer.tokenize(sent_a)

    tokens_b = None
    if sent_b:
        tokens_b = tokenizer.tokenize(sent_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens =  ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    if tokens_b:
        tokens = tokens + tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_length - len(input_ids))

    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids

def tok2int_list(src_list, tokenizer, max_seq_length):
    inp_padding = list()
    msk_padding = list()
    seg_padding = list()
    for step, sent in enumerate(src_list):
        input_ids, input_mask, input_seg = tok2int_sent(sent, tokenizer, max_seq_length)
        inp_padding.append(input_ids)
        msk_padding.append(input_mask)
        seg_padding.append(input_seg)
    return inp_padding, msk_padding, seg_padding

tokenizer = BertTokenizerRepo.from_pretrained(args['bert_pretrain'], do_lower_case=False)

inp, msk, seg = tok2int_list(test_inputs, tokenizer, args['max_len'])

inp = Variable(torch.LongTensor(inp))
msk = Variable(torch.LongTensor(msk))
seg = Variable(torch.LongTensor(seg))

if args['cuda']:
    inp = inp.cuda()
    msk = msk.cuda()
    seg = seg.cuda()

# Tokenizer could be replaced by this
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(args['bert_pretrain'], do_lower_case=False)

encodings =  tokenizer(
    test_inputs,
    padding='max_length',
    truncation='longest_first',
    max_length=args['max_len'],
    return_token_type_ids=True,
    return_attention_mask=True,
    return_tensors='pt',
)

inp = encodings['input_ids']
msk = encodings['attention_mask']
seg = encodings['token_type_ids']

if args['cuda']:
    inp = inp.cuda()
    msk = msk.cuda()
    seg = seg.cuda()

from sentence_retrieval_model import sentence_retrieval_model

model = sentence_retrieval_model(args)
model.load_state_dict(torch.load(args['checkpoint'], map_location=torch.device('cpu'))['model'])

if args['cuda']:
    model = model.cuda()

import numpy as np

model.eval()

with torch.no_grad():
    probs = model(inp, msk, seg).tolist()
    
print(len(probs), probs)
    
# This is the absolure error (rounded to 4 decimal places) between this notebook and the repo's execution
np.abs((np.array(probs) - np.array([e[3] for t in test_claim for e in t['evidence']])).round(4))

hand_inputs = [
    ['Johnny Depp is an actor.','Depp has acted on many movies.']
]
encodings = tokenizer(
    hand_inputs,
    padding='max_length',
    truncation='longest_first',
    max_length=args['max_len'],
    return_token_type_ids=True,
    return_attention_mask=True,
    return_tensors='pt',
)

inp_t = encodings['input_ids']
msk_t = encodings['attention_mask']
seg_t = encodings['token_type_ids']

if args['cuda']:
    inp_t = inp_t.cuda()
    msk_t = msk_t.cuda()
    seg_t = seg_t.cuda()
    
model.eval()

with torch.no_grad():
    probs = model(inp_t, msk_t, seg_t).tolist()
probs

#[0.9979841709136963]