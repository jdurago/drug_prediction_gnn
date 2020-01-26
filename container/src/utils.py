import matplotlib.pyplot as plt
import networkx as nx
from dgl.data.chem import smiles_to_bigraph, BaseAtomFeaturizer, CanonicalAtomFeaturizer

def smiles_to_draw_networkx(smiles_inp: str):
    mol = smiles_to_bigraph(smiles_inp)
    nx_mol = mol.to_networkx()
    plt.figure()
    nx.draw_networkx(nx_mol, pos=nx.spring_layout(nx_mol), with_labels=True)


def smiles_to_dgl_graph(smiles_str: str, node_featurizer: BaseAtomFeaturizer = CanonicalAtomFeaturizer()):
    return smiles_to_bigraph(smiles_str, atom_featurizer=node_featurizer)


