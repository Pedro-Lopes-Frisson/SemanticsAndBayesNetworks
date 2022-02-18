# encoding: utf8
#################################################################
#   Name                NMEC
#   Vicente Costa       98515
#   Joao Borges         98155
#   Filipe Gonçalves    98083
#   Gonçalo Machado     98359
#   Catarina Oliveira   98292
#
##################################################################

from semantic_network import *
from bayes_net import *
from collections import Counter


class MySemNet(SemanticNetwork):
    def __init__(self):
        SemanticNetwork.__init__(self)
        # IMPLEMENT HERE (if needed)
        self.conf = lambda n, T: n / (2 * T) + (1 - (n / (2 * T))) * (
            1 - (0.95 ** n)
        ) * 0.95 ** (T - n)

    def source_confidence(self, user):
        correct = 0
        wrong = 0

        user_assoc_one = {
            (d.relation.name, d.relation.entity1): d.relation.entity2
            for d in self.declarations
            if d.user == user and isinstance(d.relation, AssocOne)
        }

        assoc_one_all = {}
        for d in self.declarations:
            for k in user_assoc_one.keys():
                if (
                    isinstance(d.relation, AssocOne)
                    and (d.relation.name, d.relation.entity1) == k
                ):
                    try:
                        assoc_one_all[k].append(d.relation.entity2)
                    except KeyError as j:
                        assoc_one_all[k] = [d.relation.entity2]

        assoc_one_all_counters = {k: Counter(v) for k, v in assoc_one_all.items()}
        assoc_one_all_most_commons = {}

        # draws
        for k, c in assoc_one_all_counters.items():
            mc = c.most_common(1)[0]
            """
            assoc_one_all_most_commons[k] = [mc]
            for c_ in c.most_common():
                if mc[1] == c_[1] and c_ != mc:
                    assoc_one_all_most_commons[k].append(c_)
            """
            assoc_one_all_most_commons[k] = [
                c_ for c_ in c.most_common() if c_[1] == mc[1]
            ]

        # correct or wrong
        for k, v in user_assoc_one.items():
            e2 = [entry[0] for entry in assoc_one_all_most_commons[k]]
            if v in e2:
                correct += 1
            else:
                wrong += 1
                """
            found = False
            for entry in assoc_one_all_most_commons[k]:
                if v == entry[0]:
                    correct += 1
                    found = True

            if not found:
                wrong +=1
                """
        return (1 - (0.75 ** correct)) * 0.75 ** wrong

    def query_with_confidence(self, entity, assoc):
        pds = [
            d
            for d in self.query_local(e1=entity)
            if isinstance(d.relation, (Member, Subtype))
        ]  # pds

        pais = [d.relation.entity2 for d in pds] # pais
        pds_assoc = {}

        local_assoc = self.query_local(e1=entity, relname=assoc) # local assocs de entity
        e2 = [d.relation.entity2 for d in local_assoc if isinstance(d.relation, AssocOne)] # so o e2 local assocOnes
        #p2 = [d for d in self.declarations if isinstance(d.relation, Subtype) and d.relation.entity1 == entity]

        dic_extend = {}

        for e in pais:
            for k, v in self.query_with_confidence(entity=e, assoc=assoc).items():
                if k in dic_extend.keys():
                    dic_extend[k] += v
                else:
                    dic_extend[k] = v

        for k,v in dic_extend.items():
            dic_extend[k] /= len(pais)

        n_s = Counter(e2) # counter localassocon

        T = len(local_assoc)
        for k, n in n_s.most_common():  # contagens
            if k not in pds_assoc.keys():
                pds_assoc[k] = self.conf(n, T) # local

        # 1

        if len(dic_extend) == 0:
            return pds_assoc

        # 2
        if len(local_assoc) == 0:
            return {k: v * 0.9 for k,v in dic_extend.items()}



        # 3 ponto
        keylist = set(list(dic_extend.keys()) + list(pds_assoc.keys()))

        for k in keylist:
            if k not in pds_assoc.keys() and k in dic_extend.keys():
                pds_assoc[k] = dic_extend[k] * 0.1
            elif k not in dic_extend.keys() and k in pds_assoc.keys():
                pds_assoc[k] = pds_assoc[k] * 0.9
            else:
                pds_assoc[k] = (pds_assoc[k] * 0.9) + ( dic_extend[k] ) * 0.1

        return pds_assoc



class MyBN(BayesNet):
    def __init__(self):
        BayesNet.__init__(self)
        # IMPLEMENT HERE (if needed)

    def individual_probabilities(self):
        self.result = {}
        # first free var
        for v, mother in self.dependencies.items():
            if frozenset() in mother:
                self.result[v] = self.individual_probabilities_rec(v, mother)

        for v, mother in self.dependencies.items():
            if frozenset() not in mother:
                self.result[v] = self.individual_probabilities_rec(v, mother)

        return self.result

    def individual_probabilities_rec(self, var, value):
        if frozenset() in value:
            return list(value.values())[0]
        else:
            # values is a dict
            v_final = 0
            for fs, value in value.items():
                ls_fs = list(fs)
                if len(ls_fs) == 1:
                    for tpl in ls_fs:
                        if tpl[0] in self.result:
                            # get from cache
                            v =  self.result[tpl[0]]
                        else:
                            v = self.individual_probabilities_rec(tpl[0], self.dependencies[tpl[0]])
                        if not tpl[1]:
                            v = 1 - v
                        v_final += v * value
                else:
                    int_v   = 1
                    for tpl in ls_fs:
                        if tpl[0] in self.result:
                            # get from cache
                            v =  self.result[tpl[0]]
                        else:
                            v = self.individual_probabilities_rec(tpl[0], self.dependencies[tpl[0]])
                        if not tpl[1]:
                            v = 1 - v
                        int_v *= v
                        #print(f"{v}", end=' * ')
                    v_final += value * int_v
                    #print(f"{value} +", end='')

            return v_final

        return 1000
