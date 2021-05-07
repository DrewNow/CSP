""" - Read precalculated atom2vec vectors of elements
    - Remove compositions with missing elements (for which vectors aren't calculated)
    - Embed compositions as arrays of atomic vectors
    - Add padding
    
    Return padded arrays
     """

import numpy as np
import tensorflow as tf

class Embedding:

    def __init__(self, data, indexfile='atoms_AE_index.txt',
                 vectorsfile='atoms_AE_vec.txt'):

        self.data = data        
        self.index = [int(i) for i in open(indexfile, 'r').readlines()]
        self.vectors = [[float(i) for i in line.split()] for \
            line in open(vectorsfile, 'r').readlines()]
        self.ELEMENTS = tuple(
            'H|He|'
            'Li|Be|B|C|N|O|F|Ne|'
            'Na|Mg|Al|Si|P|S|Cl|Ar|'
            'K|Ca|Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|As|Se|Br|Kr|'
            'Rb|Sr|Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|I|Xe|'
            'Cs|Ba|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Po|At|Rn|'
            'Fr|Ra|Ac|Th|Pa|U|Np|Pu|Am|Cm|Bk|Cf|Es|Fm|Md|No|Lr|Rf|Db|Sg|Bh|Hs|Mt|Ds|Rg'.split('|')
        ) 

        self.found_elements = np.array(self.ELEMENTS)[self.index]
        self.missing_elements =  set(self.ELEMENTS).difference(set(self.found_elements))
        self.a2v = {element: vector for element, vector in zip(self.found_elements, self.vectors)}
        self.maxl = max(list(map(len, self.data['composition'])))
        self.d_model = len(self.vectors[0])

    def is_missing(self, composition):
        return set(composition).isdisjoint(self.missing_elements)

    def remove_missing_elements(self, name='composition'):
        self.data['is_missing'] = list(map(self.is_missing, self.data[name].values))
        df = self.data[self.data['is_missing']]
        return df.drop(columns=['is_missing'])

    def atom2vec(self, composition):
        length = len(composition)
        array = np.asarray([self.a2v[e] for e in composition])

        if length == self.maxl:
            return array
        else:
            return np.vstack([array, np.zeros((length - self.maxl, self.d_model))])

    def call(self):
        df = self.remove_missing_elements()
        df['compositions_vectors'] = list(map(self.atom2vec, df['composition'])) 
        return df, tf.convert_to_tensor(df['compositions_vectors'].values), self.d_model, self.maxl