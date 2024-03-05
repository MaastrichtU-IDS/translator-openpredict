import csv
from io import StringIO

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def prepareFASTA(infile="data/download/drugbank_targets.csv", outfile="data/download/drugbank_targets.fasta"):
    seqs = []
    with open(infile) as fin:
        for line in csv.DictReader(fin):
            fasta_io = StringIO(line["seq"])
            records = SeqIO.parse(fasta_io, "fasta")

            for rec in records:
                r = SeqRecord(Seq(rec.seq), id=line["target"], name="", description="")
                seqs.append(r)

            fasta_io.close()
            # print(r)
    SeqIO.write(seqs, outfile, "fasta")


def prepareSMILES(
    infile="data/download/drugbank_drugs.csv",
    smiles_file="data/download/drugbank_smiles.txt",
    smiles_drug_file="data/download/drugbank_drug_smiles.csv",
):
    unique_smiles = {}
    other_smiles = {}

    with open(smiles_file, "w", newline="\n") as out:
        writer = csv.DictWriter(out, fieldnames=["smiles"])

        with open(infile) as in_f:
            for row in csv.DictReader(in_f):
                drug = row["drug"]
                smiles = row["smiles"]
                if smiles not in unique_smiles:
                    unique_smiles[smiles] = drug
                    writer.writerow({"smiles": row["smiles"]})
                else:
                    if smiles not in other_smiles:
                        other_smiles[smiles] = [drug]
                    else:
                        drug_list = other_smiles[smiles]
                        drug_list.append(drug)
                        other_smiles[smiles] = drug_list

    # write the smiles to drug csv
    with open(smiles_drug_file, "w", newline="\n") as out:
        writer = csv.DictWriter(out, fieldnames=["smiles", "drug", "drugs"])
        for smiles, drug in unique_smiles.items():
            other = []
            if smiles in other_smiles:
                other = other_smiles[smiles]
            others = ";".join(other)
            writer.writerow({"smiles": smiles, "drug": drug, "drugs": others})


prepareFASTA()
prepareSMILES()
