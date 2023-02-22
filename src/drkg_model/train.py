
import pandas as pd 
import numpy as np

DRKG_PATH = "/Users/elifozkan/Desktop/translator-openpredict/data/drkg/embed/drkg.tsv"
drkg_data = pd.read_table(DRKG_PATH, sep = "\t")
entity2src = pd.read_table("/Users/elifozkan/Desktop/translator-openpredict/data/drkg/entity2src.tsv",on_bad_lines='skip')

entity2src.columns = ["Subject", "Source"]

GNBR = '[GNBR] Data extracted from biomedical texts see https://www.ncbi.nlm.nih.gov/pubmed/29490008/'
IntAct = '[IntAct] https://www.ebi.ac.uk/intact/'
BibSource = '[Bibliograpic sources] Extracted from the paper Zhou, Y., Hou, Y., Shen, J. et al. Network-based drug repurposing for novel coronavirus 2019-nCoV/SARS-CoV-2. Cell Discov 6, 14 (2020).'
Hetionet = '[Hetionet] Biomedical knowledge graph https://het.io/about/'


GNBR_entities = entity2src.loc[entity2src['Subject'] == GNBR]
IntAct_entities = entity2src.loc[entity2src['Subject'] == IntAct]
BibSource_entities = entity2src.loc[entity2src['Subject'] == BibSource]
Hetionet_entities = entity2src.loc[entity2src['Subject'] == Hetionet]

print("# extracted entities : " , len(Hetionet_entities) + len(GNBR_entities) + len(IntAct_entities)+ len(BibSource_entities))
print(entity2src['Source'].nunique())

