import os
import time
from datetime import datetime
import shutil
import pathlib

import itertools
import math
import numpy as np
import pandas as pd
import networkx as nx
# from oddt import toolkit
# from oddt import fingerprints

# import numbers
# import random
# from sklearn import model_selection, tree, ensemble, svm, linear_model, neighbors, metrics
# from sklearn.model_selection import GroupKFold, StratifiedKFold
# from joblib import dump
# from rdflib import Graph, URIRef, Literal, RDF, ConjunctiveGraph, Namespace


### TODO: Do not need to be implemented, TO REMOVE

def evaluate(train_df, test_df, clf):
    """Evaluate the trained classifier
    
    :param train_df: Train dataframe
    :param test_df: Test dataframe
    :param clf: Classifier
    :return: Scores
    """
    features = list(train_df.columns.difference(['Drug','Disease','Class']))
    X_test =  test_df[features]
    y_test = test_df['Class']

    # https://scikit-learn.org/stable/modules/model_evaluation.html#using-multiple-metric-evaluation
    scoring = ['precision', 'recall', 'accuracy', 'roc_auc', 'f1', 'average_precision']
    
    # TODO: check changes here
    # scorers, multimetric = metrics.scorer._check_multimetric_scoring(clf, scoring=scoring)
    # AttributeError: module 'sklearn.metrics' has no attribute 'scorer'
    # scorers, multimetric = metrics.get_scorer._check_multimetric_scoring(clf, scoring=scoring)
    # AttributeError: 'function' object has no attribute '_check_multimetric_scoring'
    scorers = {}
    # for scorer in scoring:
    #     scorers[scorer] = metrics.get_scorer(scorer)
    
    # scores = multimetric_score(clf, X_test, y_test, scorers)
    return scores

def grapDistance(ppi, target1, target2):
    """Get the shortest path between two proteins in the PPI network
    
    :param ppi: dictonary that contains distance of PPI 
    :param target1: first protein name
    :param target2: second protein name
    :return: the shortest path between two proteins in the PPI network
    """
    maxValue = 9999
    if target1 not in ppi:
        return maxValue
    else:
        if target2 not in ppi[target1]:
            return maxValue
        else:
            return ppi[target1][target2]

def tanimoto_score(fp1, fp2):
    """Compute the Tanimoto score
    
    :param fp1: first param
    :param fp2: second param
    :return: the tanimoto score of the 2 params
    """
    return np.sum(fp1 &  fp2) / np.sum(fp1 | fp2)

def fasta2seq(lines):
    """Convert FASTA format to sequences
    
    :param lines: FASTA lines
    :return: sequence lines
    """
    lines = lines[lines.index('\n')+1:]
    lines =lines.replace('\n','')
    return lines

def cosine_similarity(a,b):
    """Compute the cosine similarity
    
    :param a: first param
    :param b: second param
    :return: cosine similarity
    """
    return  np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def generate_feature():
    """Generate feature for drug-disease models
    
    :return: Features
    """
    time_start = datetime.now()

    # later: complete starts from the /data/sparql folder
    # Download SPARQL results as CSV: 
    # arguments: [ "-H", "Accept: text/csv","--data-urlencode", "query@$(inputs.abs_path)data/sparql/$(inputs.input)" , "$(inputs.sparql_endpoint)", "-o", "$(runtime.outdir)/$(inputs.output)"]
    # drug_target = 'drugbank-drug-target.rq'
    # target_seq = 'drugbank-target-seq.rq'
    # drug_goa = 'drugbank-drug-goa.rq'
    # drug_smiles = 'drugbank-drug-smiles.rq'
    # drug_se = 'drugbank-sider-sideeffects.rq'

    # The results of those SPARQL queries are availabe as CSV in data/input/
    # Start from the CSV at the moment

    temp = "data/tmp/"
    abs_path = str(pathlib.Path().absolute()) + "/"
    print(abs_path)
    input_folder = "data/input/"
    features_folder = "data/features/"

    # drug_target = 'drugbank-drug-target.rq'
    # target_seq = 'drugbank-target-seq.rq'
    # drug_goa = 'drugbank-drug-goa.rq'
    # drug_smiles = 'drugbank-drug-smiles.rq'
    # drug_se = 'drugbank-sider-sideeffects.rq'
    # drug_ppi = 'human-interactome.rq'

    drug_target = pd.read_csv(os.path.join(input_folder, 'drugbank-drug-target.csv'))
    target_seq = pd.read_csv(os.path.join(input_folder, 'drugbank-target-seq.csv'))
    drug_goa = pd.read_csv(os.path.join(input_folder, 'drugbank-drug-goa.csv'))
    drug_smiles = pd.read_csv(os.path.join(input_folder, 'drugbank-drug-smiles.csv'))
    drug_se = pd.read_csv(os.path.join(input_folder, 'drugbank-sider-se.csv'))
    
    drug_ind = os.path.join(input_folder, 'openpredict-omim-drug.csv')
    disease_mesh = os.path.join(input_folder, 'omim-disease-mesh.csv')
    disease_hpo = os.path.join(input_folder, 'omim-disease-hpo.csv')

    print(drug_smiles.head())

    print ("%d drugs have all Target feature "%len(  drug_target.drugid.unique()))
    print ("%d drugs have Target GOA feature "%len( drug_goa.drugid.unique()))
    print ("%d drugs have Fingerprint feature "%len(  drug_smiles.drugid.unique()))
    print ("%d drugs have Sideeffect feature "%len( drug_se.drugid.unique()))
    drug_target_seq = drug_target.merge(target_seq, on= ['geneid'])
    print ("%d drugs have Target SEQ feature "%len( drug_target_seq.drugid.unique()))

    a=drug_goa['drugid'].unique()
    b=drug_target['drugid'].unique()
    c=drug_smiles['drugid'].unique()
    d=drug_se['drugid'].unique()
    commonDrugs= set(a).intersection(b).intersection(c).intersection(d)
    print (len(a),len(b),len(c),len(d))
    print (len(commonDrugs))
    
    print(drug_se.head())

    print('Drug side effect similarity 🤮')
    print('calculating Jaccard coefficient based on drug sideefects')
    os.makedirs(os.path.join(temp, 'features'), exist_ok=True)
    inter_folder= os.path.join(abs_path,'data/intermediate')
    if os.path.isdir(inter_folder):
        shutil.rmtree(inter_folder)    
    os.makedirs(inter_folder, exist_ok=True)

    os.chdir(abs_path)


    drugSEDict = {k: g["umlsid"].tolist() for k,g in drug_se.groupby("drugid")}
    scores = []

    for comb in itertools.combinations(commonDrugs,2):
        drug1 =comb[0]
        drug2 =comb[1]

        sideeffects1 = drugSEDict[drug1]
        sideeffects2 = drugSEDict[drug2]
        c = set(sideeffects1).intersection(sideeffects2)
        u = set(sideeffects1).union(sideeffects2)
        score = len(c)/float(len(u))
        scores.append([drug1, drug2, score])

    drug_se_df = pd.DataFrame(scores, columns =['Drug1','Drug2','SE-SIM'])

    print(drug_se_df.head())

    drug_se_df.to_csv(os.path.join(temp,'features/drugs-se-sim.csv'), index=False)

    drug_ppi = 'data/input/human-interactome.csv'
    
    print('PPI based drug-drug similarity 💉 💊')
    print('calculate distance between drugs on protein-protein interaction network')
    G = nx.Graph()
    with open(drug_ppi) as ppiFile: # human PPI network
        next(ppiFile) # skip first line
        drugs=set()
        for line in ppiFile:
            line=line.replace("'","").strip().split(',')
            G.add_edge(line[0],line[1])

    ppi = nx.shortest_path_length(G)

    drug_targetlist = {k: g["geneid"].tolist() for k,g in drug_target.groupby("drugid")}
    values = []

    print('calculate PPI-based pairwise drug similarity (Closeness) 👫')
    # First distances between proteins were transformed to similarity values using the formula described in Perlman et al (2011)
    # A, b were chosen according to Perlman et al (2011) to be 0.9 × e and 1, respectively.
    # Self similarity was assigned a value of 1.

    # For drugs similarities, maximal values between the two lists of associated genes were averaged 
    # (taking into account both sides for symmetry).

    A = 0.9
    b = 1
    for comb in itertools.combinations(commonDrugs,2) :
        drug1 = comb[0]
        drug2 = comb[1]
        if not(drug1 in drug_targetlist and drug2 in drug_targetlist) : continue
        targetList1 = drug_targetlist[drug1]
        targetList2 = drug_targetlist[drug2]
        allscores =[]
        for target1 in sorted(targetList1):
            genescores = []
            for target2 in sorted(targetList2):
                target1 =str(target1)
                target2 =str(target2)    
                if target1 == target2:
                    score=1.0
                else:
                    score = A*math.exp(-b* grapDistance(ppi, target1, target2))
                genescores.append(score)
        # add maximal values between the two lists of associated genes 
        allscores.append(max(genescores))
        if len(allscores) ==0: continue
        #average the maximal scores 
        maxScore =np.mean(allscores)
        if maxScore >= 0:
            values.append([drug1, drug2, maxScore])

    drug_ppi_df = pd.DataFrame(values, columns =['Drug1','Drug2','PPI-SIM'])

    print(drug_ppi_df.head())

    drug_ppi_df.to_csv(os.path.join(temp,'features/drugs-ppi-sim.csv'), index=False)

    print('Drug fingerprint similarity 🖕')
    print('calculating MACS based fingerprint (substructure) similarity')

    drug_smiles = drug_smiles[drug_smiles.drugid.isin(commonDrugs)]
    print(drug_smiles.head())

    #Create a dictionary of chemicals to be compared:
    input_dict = dict()
    for index,line in drug_smiles.iterrows():
        id = line['drugid']
        
        smiles = line['smiles']
        mol = toolkit.readstring(format='smiles',string=smiles)
        fp =mol.calcfp(fptype='MACCS').raw
        input_dict[id] = fp

    sim_values=[]
    for chemical1, chemical2 in itertools.combinations(input_dict.keys(),2):
        TC= tanimoto_score(input_dict[chemical1], input_dict[chemical2])
        if chemical1 != chemical2:
            sim_values.append([chemical1, chemical2, TC])

    chem_sim_df = pd.DataFrame(sim_values, columns=['Drug1','Drug2','TC'])
    print(chem_sim_df.head())
    chem_sim_df.to_csv(os.path.join(temp,'features/drugs-fingerprint-sim.csv'), index=False)

    print('Drug target sequence similarity 🧬')
    print('Calculation of SmithWaterman sequence alignment scores')

    target_seq.seq =target_seq.seq.map(fasta2seq)
    target_seq = target_seq[target_seq.geneid.isin(drug_target.geneid)]
    print(target_seq.head())

    target_seq_file=os.path.join(abs_path,"data/intermediate/drugbank-target-seq-trimmed.tab")
    target_seq.to_csv(target_seq_file,'\t',index=False,header=None)
    target_seq_sim_file= os.path.join(abs_path, "data/intermediate/target-target-seq-sim-biojava.tab")

    # TODO: run Java 
    os.system('java -cp .:lib/smithwaterman.jar:lib/biojava-alignment-4.0.0.jar:lib/biojava-core-4.0.0.jar:lib/slf4j-api-1.7.10.jar biojava.targetseq.CalcLocalAlign ' + target_seq_file + ' > ' + target_seq_sim_file)

    targetSeqSim=dict()
    with open(target_seq_sim_file) as tarSimfile:
        for row in tarSimfile:
            row = row.strip().split("\t")
            t1 =row[0]
            t2 = row[1]
            sim = float(row[2])
            targetSeqSim[(t1,t2)]=sim
            targetSeqSim[(t2,t1)]=sim

    drug_targetlist = {k: g["geneid"].tolist() for k,g in drug_target_seq.groupby("drugid")}
    values = []

    for comb in itertools.combinations(commonDrugs,2) :
        drug1 = comb[0]
        drug2 = comb[1]
        if not(drug1 in drug_targetlist and drug2 in drug_targetlist) : continue
        targetList1 = drug_targetlist[drug1]
        targetList2 = drug_targetlist[drug2]
        allscores =[]
        for target1 in sorted(targetList1):
            genescores = []
            for target2 in sorted(targetList2):
                target1 =str(target1)
                target2 =str(target2)    
                if target1 == target2:
                    score=1.0
                else:
                    score = targetSeqSim[(target1,target2)] / (math.sqrt(targetSeqSim[(target1,target1)]) * math.sqrt(targetSeqSim[(target2,target2)]))
                genescores.append(score)
        # add maximal values between the two lists of associated genes 
        allscores.append(max(genescores))
        if len(allscores) ==0: continue
        #average the maximal scores 
        maxScore =np.mean(allscores)
        values.append([drug1, drug2, maxScore])

    drug_seq_df = pd.DataFrame(values, columns =['Drug1','Drug2','TARGETSEQ-SIM'])
    print(drug_seq_df.head())
    drug_seq_df.to_csv(os.path.join(temp,'features/drugs-target-seq-sim.csv'), index=False)

    print('GO based drug-drug similarity 🧬 💉 💊')

    drug_goa.drugid = drug_goa.drugid.map(lambda d: 'http://purl.obolibrary.org/obo/'+d)
    drug_goa.to_csv(os.path.join(abs_path,'data/intermediate/drug_goa.txt'),sep='\t', header=False, index=False)

    # Cleaning GO annotations
    rows = []
    for comb in itertools.combinations(commonDrugs,2):
        t1=comb[0]
        t2=comb[1]
        rows.append(['http://purl.obolibrary.org/obo/'+str(t1),'http://purl.obolibrary.org/obo/'+str(t2)])

    drug_query_df = pd.DataFrame(rows, columns =['Drug1','Drug2'])
    drug_query_df.to_csv(os.path.join(abs_path,'data/intermediate/drug.gene.go.query'),sep='\t', header=False, index=False)

    # TODO: unzip GO ontology
    # ! if [ ! -f 'data/ontology/go.owl' ]; then gunzip 'data/ontology/go.owl'; fi

    # TODO: run Java
    # Run the semantic relatedness library with given query and anotation file it will produce a file named: gene.go.sim.out
    os.system('java -jar lib/sml-toolkit-0.9.jar -t sm -xmlconf data/conf/sml.gene.go.conf')

    go_sim_df = pd.read_csv(os.path.join(abs_path,'data/intermediate/drug.gene.go.sim.out'),sep='\t')
    print(go_sim_df.head())

    go_sim_df.rename(columns={'e1':'Drug1','e2':'Drug2','bma':'GO-SIM'}, inplace=True)

    go_sim_df.Drug1 = go_sim_df.Drug1.str.replace('http://purl.obolibrary.org/obo/','')
    go_sim_df.Drug2 = go_sim_df.Drug2.str.replace('http://purl.obolibrary.org/obo/','')
    print(go_sim_df.head())

    go_sim_df.to_csv(os.path.join(temp, 'features/drugs-target-go-sim.csv'))

    print('Disease Phenotype Similarity')
    print('MESH term based Similarity')
    predict_df = pd.read_csv(drug_ind)
    predict_df.head()

    predict_df.rename(columns={'drugid':'Drug','omimid':'Disease'}, inplace=True)

    gold_diseases = set( predict_df.Disease.unique())
    print ('Gold std. diseases',len(gold_diseases))

    mesh_ann = {}
    allmeshterm = []
    with open(disease_mesh) as meshfile:
        next(meshfile)
        for line in meshfile:
            line = line.strip().split(',')
            if len(line) != 2: continue
            di = line[0]
            mesh = line[1].split(',')
            mesh_ann[di]=mesh
            allmeshterm.extend(mesh)

    vocabulary = list(set(allmeshterm))
    print('Mesh term count: ' + str(len(vocabulary)))

    # create a co-occurrence matrix
    co_mat = np.zeros((len(mesh_ann),len(vocabulary)))

    commonDiseases = mesh_ann.keys()
    mesh2id= { di:i for i,di in enumerate(mesh_ann.keys())}
    # fill in the co-occurrence matrix
    for key in mesh_ann:
        annotations = mesh_ann[key]
        col_index = [vocabulary.index(a) for a in annotations]
        co_mat[mesh2id[key],col_index] =1

    values = []
    # calculate cosine similarity between diseases using mesh annotation vector
    for comb in itertools.combinations(commonDiseases,2) :
        disease1 = comb[0]
        disease2 = comb[1]
        sim = cosine_similarity(co_mat[mesh2id[disease1],:], co_mat[mesh2id[disease2],:])
        values.append([disease1, disease2, sim])

    disease_pheno_df = pd.DataFrame(values, columns =['Disease1','Disease2','PHENO-SIM'])
    print(disease_pheno_df.head())
    disease_pheno_df.to_csv(os.path.join(temp, 'features/diseases-pheno-sim.csv'),index=False)

    print('HPO based disease-disease similarity 🦠 🧫')
    # reading the hpo annotation file taken from compbio.charite.de/jennkins/jobs/hpo.annotations/
    disease_hpo = pd.read_csv(disease_hpo)
    print(disease_hpo.head())

    disease_hpo.rename(columns={'diseaseid':'Disease','hpoid':'HPO'}, inplace=True)
    disease_hpo.HPO= disease_hpo.HPO.str.replace('hpo','hp')
    print(disease_hpo.head())

    diseasesWithFeatures= set(disease_hpo.Disease.unique()).intersection( gold_diseases )
    print(len(diseasesWithFeatures))
    rows = []
    for comb in itertools.combinations(diseasesWithFeatures,2):
        t1=comb[0]
        t2=comb[1]
        rows.append(['omim:'+str(t1),'omim:'+str(t2)])

    disease_hpo["Disease"]=disease_hpo["Disease"].map(lambda d: 'omim:'+str(d))
    disease_hpo.to_csv(os.path.join(abs_path,'data/intermediate/disease_hpo.txt'), sep='\t', header=False, index=False)

    disease_query_df = pd.DataFrame(rows, columns =['Disease1','Disease2'])
    disease_query_df.to_csv(os.path.join(abs_path, 'data/intermediate/hpo.sml.omim.query'), sep='\t', header=False, index=False)

    # TODO: unzip HPO ontology
    # ! if [ ! -f 'data/ontology/hpo.owl' ]; then gunzip 'data/ontology/hpo.owl'; fi

    ### run the semantic relatedness library with given query and anotation file it will produce a file named: hpo.sim.out
    os.system('java -jar lib/sml-toolkit-0.9.jar -t sm -xmlconf data/conf/sml.omim.hpo.conf')

    hpo_sim_df = pd.read_csv(os.path.join(abs_path,'data/intermediate/omim.hpo.sim.out'),sep='\t')

    print(hpo_sim_df.head())
    
    hpo_sim_df.rename(columns={'e1':'Disease1','e2':'Disease2','bma':'HPO-SIM'}, inplace=True)
    hpo_sim_df.Disease1 = hpo_sim_df.Disease1.str.replace('omim:','')
    hpo_sim_df.Disease2 = hpo_sim_df.Disease2.str.replace('omim:','')
    print(hpo_sim_df.head())

    hpo_sim_df.to_csv(os.path.join(temp,'features/diseases-hpo-sim.csv'), index=False)

    # # Feature extraction (Best Combined similarity)
    # print('\nFeature extraction ⛏️')
    # knownDrugDisease = pairs_train[classes_train==1]
    # time_pairs_train = datetime.now()
    # print('Pairs train runtime 🕓  ' + str(time_pairs_train - time_start))
    # print('\nCalculate the combined similarity of the training pairs 🏳️‍🌈')
    # train_df, test_df = calculateCombinedSimilarity(pairs_train, pairs_test, classes_train, classes_test, drug_df, disease_df, knownDrugDisease)
    # time_calculate_similarity = datetime.now()
    # print('CalculateCombinedSimilarity runtime 🕓  ' + str(time_calculate_similarity - time_pairs_train))

    # # Model Training, get classifier (clf)
    # print('\nModel training, getting the classifier 🏃')
    # n_seed = 100
    # clf = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, random_state=n_seed) 
    # clf = trainModel(train_df, clf)
    # time_training = datetime.now()
    # print('Model training runtime 🕕  ' + str(time_training - time_calculate_similarity))

    # # Evaluation of the trained model
    # print('\nRunning evaluation of the model 📝')
    # scores = evaluate(train_df, test_df, clf)
    # time_evaluate = datetime.now()
    # print('Evaluation runtime 🕗  ' + str(time_evaluate - time_training))

    # # About 3min to run on a laptop
    # print("\nTest results 🏆")
    # print(scores)

    # print('\nStore the model in a .joblib file 💾')
    # dump(clf, 'data/models/drug_disease_model.joblib')
    # See skikit docs: https://scikit-learn.org/stable/modules/model_persistence.html

    print('Complete runtime 🕛  ' + str(datetime.now() - time_start))
    return "yes"