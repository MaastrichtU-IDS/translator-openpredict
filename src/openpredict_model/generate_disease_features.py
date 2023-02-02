import argparse
import itertools
import os

import numpy as np
import pandas as pd

from openpredict.config import settings

# Standalone script to generate diseases features

def fasta2seq(lines):
    lines = lines[lines.index('\n')+1:]
    lines = lines.replace('\n', '')
    return lines

def download():
    """Download a jar file required for the processing in data/lib/"""
    if not os.path.exists(settings.OPENPREDICT_DATA_DIR / "lib" / "sml-toolkit-0.9.jar" ):
        print("sml-toolkit-0.9.jar not present, downloading it")
        try:
            os.system(f'mkdir -p data/lib')
            os.system(f"wget -q --show-progress https://repo1.maven.org/maven2/com/github/sharispe/slib-tools-sml-toolkit/0.9/slib-tools-sml-toolkit-0.9.jar -O data/lib/sml-toolkit-0.9.jar")
        except Exception as e:
            print(f"Error while downloading kgpredict: {e}")



if __name__ == "__main__":
    download()
    #cd /data
    # git clone https://github.com/fair-workflows/openpredict
    # python generateDiseaseFeatures.py -hpo data/external/phenotype_annotation_hpoteam.tab -mesh data/external/mim2mesh.tsv -di data/input/openpredict-omim-drug.csv -t ../temp -a /data/openpredict/
    # cp ../temp/features/* data/baseline_features
    parser = argparse.ArgumentParser()
    parser.add_argument('-mesh', required=True,
                        dest='mesh_annotation', help='enter path to temp path ')
    parser.add_argument('-hpo', required=True,
                        dest='hpo_annotation', help='enter path to temp path ')
    parser.add_argument('-di', required=True,
                        dest='drug_indication', help='enter path to temp path ')

    parser.add_argument('-t', required=True, dest='temp',
                        help='enter path to temp folder')
    parser.add_argument('-a', required=True, dest='absolute',
                        help='enter path to temp folder')

    args = parser.parse_args()

    temp_folder = args.temp
    abs_path = args.absolute

    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)

    if not os.path.exists(os.path.join(temp_folder, 'features')):
        os.mkdir(os.path.join(temp_folder, 'features'))
    if not os.path.exists(os.path.join(temp_folder, 'intermediate')):
        os.mkdir(os.path.join(temp_folder, 'intermediate'))

    temp_folder = os.path.abspath(temp_folder)
    predict_df = pd.read_csv(os.path.join(abs_path, args.drug_indication))
    predict_df.head()

    # reading the hpo annotation file taken from compbio.charite.de/jennkins/jobs/hpo.annotations/

    columns = ["Source", "Disease", "x0", "x1", "HPO", "x3",
               "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11"]
    disease_hpo = pd.read_csv(os.path.join(
        abs_path, args.hpo_annotation), names=columns, sep='\t')
    disease_hpo = disease_hpo[disease_hpo['Source'] == 'OMIM']
    disease_hpo = disease_hpo[disease_hpo['HPO'].str.startswith('HP')]
    disease_hpo = disease_hpo[['Disease', 'HPO']]
    print(disease_hpo.head())

    # In[142]:

    predict_df.rename(
        columns={'drugid': 'Drug', 'omimid': 'Disease'}, inplace=True)

    # In[141]:

    gold_diseases = set(predict_df.Disease.unique())
    print('Gold std. diseases', len(gold_diseases))

    # ## Disease Phenotype Similarity
    # ### MESH term based Similarity
    print("MESH term based Similarity")

    mesh_ann = {}
    allmeshterm = []
    with open(os.path.join(abs_path, args.mesh_annotation)) as meshfile:
        next(meshfile)
        for line in meshfile:
            line = line.strip().split('\t')
            #print (line)
            # if len(line) != 2: continue
            di = line[0]
            if int(di) not in gold_diseases:
                continue
            mesh = line[1:]
            mesh_ann[di] = mesh
            #print (di,':', mesh)
            allmeshterm.extend(mesh)

    # In[143]:

    vocabulary = list(set(allmeshterm))
    print('voc', len(vocabulary))

    # In[144]:

    # create a co-occurrence matrix
    co_mat = np.zeros((len(mesh_ann), len(vocabulary)))

    # In[145]:

    commonDiseases = mesh_ann.keys()
    mesh2id = {di: i for i, di in enumerate(mesh_ann.keys())}
    # fill in the co-occurrence matrix
    for key in mesh_ann:
        annotations = mesh_ann[key]
        col_index = [vocabulary.index(a) for a in annotations]
        co_mat[mesh2id[key], col_index] = 1

    # In[146]:

    def cosine_similarity(a, b):
        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

    # In[147]:

    values = []
    # calculate cosine similarity between diseases using mesh annotation vector
    for comb in itertools.combinations(commonDiseases, 2):
        disease1 = comb[0]
        disease2 = comb[1]
        sim = cosine_similarity(
            co_mat[mesh2id[disease1], :], co_mat[mesh2id[disease2], :])
        values.append([disease1, disease2, sim])

    # In[148]:

    disease_pheno_df = pd.DataFrame(
        values, columns=['Disease1', 'Disease2', 'PHENO-SIM'])

    # In[149]:

    disease_pheno_df.head()

    # In[150]:

    disease_pheno_df.to_csv(os.path.join(
        temp_folder, 'features/diseases-pheno-sim.csv'), index=False)
    print("MESH term based Similarity --  done")

    # ## HPO based disease-disease similarity

    # In[69]:

    # reading the hpo annotation file taken from compbio.charite.de/jennkins/jobs/hpo.annotations/

    # In[70]:

    print("HPO term based Similarity")

    disease_hpo.rename(
        columns={'diseaseid': 'Disease', 'hpoid': 'HPO'}, inplace=True)
    disease_hpo.HPO = disease_hpo.HPO.str.replace('hpo', 'hp')

    # In[71]:

    disease_hpo.head()

    # In[72]:

    diseasesWithFeatures = set(
        disease_hpo.Disease.unique()).intersection(gold_diseases)
    print(len(diseasesWithFeatures))
    rows = []
    for comb in itertools.combinations(diseasesWithFeatures, 2):
        t1 = comb[0]
        t2 = comb[1]
        rows.append(['omim:'+str(t1), 'omim:'+str(t2)])

    # In[73]:

    disease_hpo["Disease"] = disease_hpo["Disease"].map(
        lambda d: 'omim:'+str(d))
    disease_hpo.to_csv(os.path.join(
        abs_path, 'data/intermediate/disease_hpo.txt'), sep='\t', header=False, index=False)

    # In[74]:

    disease_query_df = pd.DataFrame(rows, columns=['Disease1', 'Disease2'])
    disease_query_df.to_csv(os.path.join(
        abs_path, 'data/intermediate/hpo.sml.omim.query'), sep='\t', header=False, index=False)

    # In[77]:

    os.chdir(abs_path)
    # run the semantic relatedness library with given query and anotation file it will produce a file named: hpo.sim.out
    os.system(
        'java -jar data/lib/sml-toolkit-0.9.jar -t sm -xmlconf data/conf/sml.omim.hpo.conf')
    # os.chdir(temp_folder)

    # In[101]:

    hpo_sim_df = pd.read_csv(os.path.join(
        abs_path, 'data/intermediate/omim.hpo.sim.out'), sep='\t')

    # In[102]:

    hpo_sim_df.head()

    # In[104]:

    hpo_sim_df.rename(
        columns={'e1': 'Disease1', 'e2': 'Disease2', 'bma': 'HPO-SIM'}, inplace=True)

    # In[106]:

    hpo_sim_df.Disease1 = hpo_sim_df.Disease1.str.replace('omim:', '')
    hpo_sim_df.Disease2 = hpo_sim_df.Disease2.str.replace('omim:', '')
    hpo_sim_df.head()

    # In[107]:

    hpo_sim_df.to_csv(os.path.join(
        temp_folder, 'features/diseases-hpo-sim.csv'), index=False)
    print("HPO term based Similarity --  done")

    # In[ ]:
