import pandas as pd
from nltk.stem.snowball import FrenchStemmer
from nltk.corpus import stopwords
import unidecode
import jellyfish
import numpy as np
from text_to_num import text2num,alpha2digit
import ast


class CleanData() : 
    def __init__(self,df_finalKG) :
            self.df_finalKG = df_finalKG

    def cleanNaN(self):
        self.df_finalKG.toponym = self.df_finalKG.toponym.apply(lambda x: None if type(x)==float else x)
        self.df_finalKG.toponym = self.df_finalKG.toponym.apply(lambda x: None if str(x).isnumeric() else x)

        self.df_finalKG.et = self.df_finalKG.et.apply(lambda x: None if type(x)==float else x)
        self.df_finalKG.et = self.df_finalKG.et.apply(lambda x: None if str(x).isnumeric() else x)

        self.df_finalKG.eg = self.df_finalKG.eg.apply(lambda x: None if type(x)==float else x)
        self.df_finalKG.eg = self.df_finalKG.eg.apply(lambda x: None if str(x).isnumeric() else x)

        self.df_finalKG.md = self.df_finalKG.md.apply(lambda x: None if type(x)==float else x)
        self.df_finalKG.md = self.df_finalKG.md.apply(lambda x: None if str(x).isnumeric() else x)
        
        return self.df_finalKG

    def removeStopwords(self, x,final_stopwords_list):
        if(x!=None):
            return ' '.join([i for i in unidecode.unidecode(x.lower()).split(" ") if i not in final_stopwords_list])
        else:
            return x
    

    def abreviation(self,x,abr,word):
        lst_word = []
        for i in x.split(" "):
            if(i!=abr):
                lst_word.append(i)
            else:
                lst_word.append(word)
        return lst_word


    def identical_stem(self,x,col_clean,col_stem):
        if(x[col_stem]!=None):
            top_word = self.df_finalKG[self.df_finalKG[col_stem]==x[col_stem]][col_clean].value_counts().index[0]
            return top_word
        else:
            return None

    def numeric_attributes(self,x):
        if(x.eg_clean_new!=None):
            if(x.eg_clean_new.split(" ")[-1].isnumeric()):
                x["attr_numeric"] = x.eg_clean_new.split(" ")[-1]
                x["eg_clean_new"] = " ".join(x.eg_clean_new.split(" ")[:-1])

        return x

    def replaceEntity(self,col,threshold): 
        lst_to_clean = list(self.df_finalKG[col])
        lst_clean = list(dict(self.df_finalKG[col].value_counts()).keys())
        dict_to_clean = dict(self.df_finalKG[col].value_counts())
        for phrase1 in lst_clean[::-1] : 
            lst_sim = []
            lst_phrase = []
            for j in list(np.where(self.df_finalKG[col].value_counts()>dict_to_clean[phrase1])[0]):
                phrase2=lst_clean[j]
                if((phrase1!=None)&(phrase2!=None)&(phrase1!=phrase2)):
                    lst_phrase1 = [word for word in phrase1.split(" ")]
                    lst_phrase2 = [word for word in phrase2.split(" ")]
                    lst_phrase1.sort()
                    lst_phrase2.sort()
                    phrase1_sort = " ".join(lst_phrase1)
                    phrase2_sort = " ".join(lst_phrase2)
                    sim = jellyfish.jaro_winkler_similarity(phrase1_sort,phrase2_sort)
                    lst_sim.append(sim)
                    lst_phrase.append(phrase2)
            indices = [i for i, x in enumerate(lst_to_clean) if x == phrase1]
            if(lst_sim!=[]):
                if(max(lst_sim)>=threshold):
                    phrase2_max = lst_phrase[lst_sim.index(max(lst_sim))]
                    if(dict_to_clean[phrase2_max]>=dict_to_clean[phrase1]):
                        for i in indices : 
                            lst_to_clean[i] = phrase2_max

        return lst_to_clean
    def removeNonFrequentEntity(self,col,threshold):
        lst_to_clean = list(self.df_finalKG[col])
        lst_to_remove = [list(self.df_finalKG[col].value_counts().index)[i] for i in list(np.where(self.df_finalKG[col].value_counts()<np.quantile(self.df_finalKG[col].value_counts(),threshold))[0])]
        for phrase1 in lst_to_remove:
            indices = [i for i, x in enumerate(lst_to_clean) if x == phrase1]
            for i in indices :
                lst_to_clean[i] = None
        return lst_to_clean
    
    def removeNonFrequentEntityAttr(self,data,col,threshold):
        lst_to_clean = list(data[col])
        lst_to_remove = [list(data[col].value_counts().index)[i] for i in list(np.where(data[col].value_counts()<np.quantile(data[col].value_counts(),threshold))[0])]
        for phrase1 in lst_to_remove:
            indices = [i for i, x in enumerate(lst_to_clean) if x == phrase1]
            for i in indices :
                lst_to_clean[i] = None
        return lst_to_clean

    def group_measure(self,x):
        sent = x.et_clean_new_3
        if((x.measure_clean_new_2!=None) & (sent!=None)):
            sent = x.measure_clean_new_2 + " " + sent
        return sent

    def group_attr(self,x):
        if(x["eg_clean_new_3"]!=None):
            if(x["attr_numeric"]!=None):
                return x["eg_clean_new_3"]+" "+x["attr_numeric"]
            else:
                return x["eg_clean_new_3"]
        else:
            return None

    def CleanDataMain(self):

        self.df_finalKG = self.cleanNaN()

        #Remove Stopwords + Abreviations
        final_stopwords_list = stopwords.words('french')
        final_stopwords_list.remove("m")
        final_stopwords_list.remove("pas")
        final_stopwords_list.remove("sur")
        final_stopwords_list.remove("dans")
        final_stopwords_list.remove("est")
        
        self.df_finalKG["toponym_clean"] = self.df_finalKG.toponym.apply(lambda x: self.removeStopwords(x,final_stopwords_list))
        self.df_finalKG["eg_clean"] = self.df_finalKG.eg.apply(lambda x: self.removeStopwords(x,final_stopwords_list))
        self.df_finalKG["et_clean"] = self.df_finalKG.et.apply(lambda x: self.removeStopwords(x,final_stopwords_list))
        self.df_finalKG["md_clean"] = self.df_finalKG.md.apply(lambda x: self.removeStopwords(x,final_stopwords_list))

        self.df_finalKG["toponym_clean"] = self.df_finalKG.toponym_clean.apply(lambda x: ' '.join(self.abreviation(x,"st","saint")) if x!=None else x)
        self.df_finalKG["toponym_clean"] = self.df_finalKG.toponym_clean.apply(lambda x: ' '.join(self.abreviation(x,"ste","sainte")) if x!=None else x)
        self.df_finalKG["toponym_clean"] = self.df_finalKG.toponym_clean.apply(lambda x: ' '.join(self.abreviation(x,"mt","mont")) if x!=None else x)

        self.df_finalKG["eg_clean"] = self.df_finalKG.eg_clean.apply(lambda x: ' '.join(self.abreviation(x,"tram","tramway")) if x!=None else x)
        self.df_finalKG["eg_clean"] = self.df_finalKG.eg_clean.apply(lambda x: ' '.join(self.abreviation(x,"fac","faculte")) if x!=None else x)
        self.df_finalKG["eg_clean"] = self.df_finalKG.eg_clean.apply(lambda x: ' '.join(self.abreviation(x,"bd","boulevard")) if x!=None else x)
        self.df_finalKG["eg_clean"] = self.df_finalKG.eg_clean.apply(lambda x: ' '.join(self.abreviation(x,"bld","boulevard")) if x!=None else x)
        self.df_finalKG["eg_clean"] = self.df_finalKG.eg_clean.apply(lambda x: ' '.join(self.abreviation(x,"blvd","boulevard")) if x!=None else x)
        self.df_finalKG["eg_clean"] = self.df_finalKG.eg_clean.apply(lambda x: ' '.join(self.abreviation(x,"bvd","boulevard")) if x!=None else x)
        self.df_finalKG["eg_clean"] = self.df_finalKG.eg_clean.apply(lambda x: ' '.join(self.abreviation(x,"av","avenue")) if x!=None else x)
        self.df_finalKG["eg_clean"] = self.df_finalKG.eg_clean.apply(lambda x: ' '.join(self.abreviation(x,"ave","avenue")) if x!=None else x)
        self.df_finalKG["eg_clean"] = self.df_finalKG.eg_clean.apply(lambda x: ' '.join(self.abreviation(x,"ch","chemin")) if x!=None else x)
        self.df_finalKG["eg_clean"] = self.df_finalKG.eg_clean.apply(lambda x: ' '.join(self.abreviation(x,"rte","route")) if x!=None else x)

        self.df_finalKG["eg_clean"] = self.df_finalKG.eg_clean.apply(lambda x: ' '.join(self.abreviation(x,"car","carrefour")) if x!=None else x)

        self.df_finalKG["et_clean"] = self.df_finalKG.et_clean.apply(lambda x: ' '.join(self.abreviation(x,"m","metres")) if x!=None else x)
        self.df_finalKG["et_clean"] = self.df_finalKG.et_clean.apply(lambda x: ' '.join(self.abreviation(x,"min","minutes")) if x!=None else x)
        self.df_finalKG["et_clean"] = self.df_finalKG.et_clean.apply(lambda x: ' '.join(self.abreviation(x,"mins","minutes")) if x!=None else x)
        self.df_finalKG["et_clean"] = self.df_finalKG.et_clean.apply(lambda x: ' '.join(self.abreviation(x,"mn","minutes")) if x!=None else x)
        self.df_finalKG["et_clean"] = self.df_finalKG.et_clean.apply(lambda x: ' '.join(self.abreviation(x,"mns","minutes")) if x!=None else x)
        self.df_finalKG["et_clean"] = self.df_finalKG.et_clean.apply(lambda x: ' '.join(self.abreviation(x,"km","kilometres")) if x!=None else x)
        self.df_finalKG["et_clean"] = self.df_finalKG.et_clean.apply(lambda x: ' '.join(self.abreviation(x,"kms","kilometres")) if x!=None else x)
        self.df_finalKG["et_clean"] = self.df_finalKG.et_clean.apply(lambda x: ' '.join(self.abreviation(x,"cur","coeur")) if x!=None else x)
        self.df_finalKG["et_clean"] = self.df_finalKG.et_clean.apply(lambda x: ' '.join(self.abreviation(x,"h","heure")) if x!=None else x)
        self.df_finalKG["et_clean"] = self.df_finalKG.et_clean.apply(lambda x: ' '.join(self.abreviation(x,"mtr","metres")) if x!=None else x)
        self.df_finalKG["et_clean"] = self.df_finalKG.et_clean.apply(lambda x: ' '.join(self.abreviation(x,"mt","metres")) if x!=None else x)
        self.df_finalKG["et_clean"] = self.df_finalKG.et_clean.apply(lambda x: ' '.join(self.abreviation(x,"mm","minutes")) if x!=None else x)

        ## Clean Entity with Stems

        stemmer = FrenchStemmer()
        self.df_finalKG["et_stem"] = self.df_finalKG.et_clean.apply(lambda x : stemmer.stem(x) if x!=None else x)
        self.df_finalKG["eg_stem"] = self.df_finalKG.eg_clean.apply(lambda x : stemmer.stem(x) if x!=None else None)

        dict_et_stem = {}
        for et_stem in self.df_finalKG.et_stem.unique() :
            if(et_stem!=None):
                top_word = self.df_finalKG[self.df_finalKG["et_stem"]==et_stem]["et_clean"].value_counts().index[0]
                dict_et_stem.update({et_stem:top_word})

        self.df_finalKG["et_clean_new"] = self.df_finalKG.et_stem.apply(lambda x : dict_et_stem[x] if x!=None else None)

        dict_eg_stem = {}
        for eg_stem in self.df_finalKG.eg_stem.unique() :
            if(eg_stem!=None):
                top_word = self.df_finalKG[self.df_finalKG["eg_stem"]==eg_stem]["eg_clean"].value_counts().index[0]
                dict_eg_stem.update({eg_stem:top_word})

        self.df_finalKG["eg_clean_new"] = self.df_finalKG.eg_stem.apply(lambda x : dict_eg_stem[x] if x!=None else None)

        ## Save and Remove Num of eg_clean_new
        self.df_finalKG["attr_numeric"] = None
        self.df_finalKG = self.df_finalKG.apply(lambda x : self.numeric_attributes(x),axis=1)

        ## CLEAN Spatial Realation
        self.df_finalKG["et_clean_new_2"] = self.replaceEntity("et_clean_new",0.9)

        self.df_finalKG["et_clean_new_3"] = self.removeNonFrequentEntity("et_clean_new_2",0.9)
        self.df_finalKG["et_clean_new_3"] = self.df_finalKG.et_clean_new_3.apply(lambda x: None if x=="" else x)

        self.df_finalKG["measure_clean_new"] =  self.df_finalKG.measure.apply(lambda x : alpha2digit(unidecode.unidecode(x.lower()), "fr") if type(x)!=float else None)

        self.df_finalKG["measure_clean_new_2"] = self.replaceEntity("measure_clean_new",0.9)

        self.df_finalKG["et_tot"] = self.df_finalKG.apply(lambda x : self.group_measure(x) ,axis=1)

        ## CLEAN Feature
        self.df_finalKG["eg_clean_new_2"] =  self.replaceEntity("eg_clean_new",0.9)

        self.df_finalKG["eg_clean_new_3"] = self.removeNonFrequentEntity("eg_clean_new_2",0.8)
        self.df_finalKG["eg_clean_new_3"] = self.df_finalKG.eg_clean_new_3.apply(lambda x: None if x=="" else x)

        self.df_finalKG["eg_clean_new_4"] = self.df_finalKG.apply(lambda x : self.group_attr(x),axis=1)
        self.df_finalKG["eg_clean_new_4"] = self.df_finalKG.eg_clean_new_4.apply(lambda x: None if x=="" else x)

        ## Clean Feature's attribtues

        self.df_finalKG.attributes = self.df_finalKG.attributes.apply(lambda x : ast.literal_eval(x))
        df_attributs = self.df_finalKG[self.df_finalKG.attributes.str.len() != 0].loc[:,["idAds","attributes"]]
        df_attributs = df_attributs.explode("attributes")
        df_attributs["attributes_clean"] = df_attributs.attributes.apply(lambda x : unidecode.unidecode(x.lower()))
        df_attributs["attributes_stem"] = df_attributs.attributes_clean.apply(lambda x : stemmer.stem(x))

        dict_attr_stem = {}
        for attr_stem in df_attributs.attributes_stem.unique() :
            if(attr_stem!=None):
                top_word = df_attributs[df_attributs["attributes_stem"]==attr_stem]["attributes_clean"].value_counts().index[0]
                dict_attr_stem.update({attr_stem:top_word})
        df_attributs["attributes_clean_final"] = df_attributs.attributes_stem.apply(lambda x : dict_attr_stem[x] if x!=None else None)

        df_attributs["attributes_clean_final_2"] = self.removeNonFrequentEntityAttr(df_attributs,"attributes_clean_final",0.7)
        df_attributs["attributes_clean_final_2"] = df_attributs.attributes_clean_final_2.apply(lambda x: None if x=="" else x)

        df_attributs = df_attributs.groupby([df_attributs.index]).agg({'attributes_clean_final_2': list})
        
        self.df_finalKG =self.df_finalKG.drop("index",axis=1)

        self.df_finalKG["attributes_clean"] = None
        for index,row in df_attributs.iterrows(): 
            self.df_finalKG.at[index,"attributes_clean"]  =  row["attributes_clean_final_2"]

        ## Clean Toponym

        self.df_finalKG["toponym_clean"]= self.df_finalKG.toponym_clean.apply(lambda x : None if type(x)==float else x)
        self.df_finalKG["toponym_clean"]= self.df_finalKG.toponym_clean.apply(lambda x : (None if x.isnumeric() else x) if x!=None else None)

        for entity in self.df_finalKG.eg_clean_new_4.unique() : 
            if(entity!=None):
                count_na = len(self.df_finalKG[(self.df_finalKG.eg_clean_new_4==entity)&(self.df_finalKG.et_clean_new_3.isna())].loc[:,["idAds"]].drop_duplicates())
                count_not_na = len(self.df_finalKG[(self.df_finalKG.eg_clean_new_4==entity)&(~self.df_finalKG.et_clean_new_3.isna())].loc[:,["idAds"]].drop_duplicates())
                if((count_not_na/(count_not_na+count_na))>0.5) : 
                    self.df_finalKG.loc[self.df_finalKG[(self.df_finalKG.eg_clean_new_4==entity)&(self.df_finalKG.et_tot.isna())].index,"et_tot"] = self.df_finalKG[(self.df_finalKG.eg_clean_new_4==entity)&(~self.df_finalKG.et_clean_new_3.isna())]["et_tot"].value_counts().index[0]

        self.df_finalKG["toponym_clean_new_2"] = self.replaceEntity("toponym_clean",0.95)
        self.df_finalKG["toponym_clean_new_2"] = self.df_finalKG.toponym_clean_new_2.apply(lambda x: None if x=="" else x)
        self.df_finalKG = self.df_finalKG.drop(self.df_finalKG[(self.df_finalKG.eg_clean_new_4.isin([None]))&(self.df_finalKG.toponym_clean_new_2.isin([None]))].index,axis=0)

        return self.df_finalKG

