
echo `pwd` > pwdfile.txt
#download kg predict drugrepurposing files
wget -q --show-progress purl.org/kgpredict -O kgpredictfiles.tar.gz
#extract kgpredict files

tar -xzvf kgpredictfiles.tar.gz  -C ./data/kgpredict_data/
rm kgpredictfiles.tar.gz

mv ./openpredict/data/kgpredict/embed/DRKG_TransE_l2_entity.npy ./openpredict/data/kgpredict/embed/entity_embeddings.npy
mv ./openpredict/data/kgpredict/embed/DRKG_TransE_l2_relation.npy ./openpredict/data/kgpredict/embed/relation_embeddings.npy

# dvc import-url https://purl.org/kgpredict data/kgpredict
