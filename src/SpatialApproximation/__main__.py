import pandas as pd
from nltk.corpus import stopwords
import pickle
import CleanData as cleandata
import GroupData as groupdata
import SpatialApproximation as sa

print("START")
dfAds = pd.read_csv("../../dataset/initial_dataset.csv",sep=';')

df_finalKG = pd.read_pickle("./data_example/dffinalKG.pkl")
df_finalKG = df_finalKG.reset_index()

new_clean = cleandata.CleanData(df_finalKG)

df_finalKG = new_clean.CleanDataMain()

new_group = groupdata.GroupData(df_finalKG)

df_finalKG, dfFeature = new_group.GroupDataMain()

attributes_rdf = df_finalKG[(~df_finalKG.toponym_clean_new_2.isna())&(~df_finalKG.attributes_clean.isna())].loc[:,["idAds","eg_clean_new_4","toponym_clean_new_2","eg_toponym","attributes_clean"]].explode("attributes_clean")
attributes_rdf = attributes_rdf.drop_duplicates()
attributes_rdf["inseeCode"] = attributes_rdf.idAds.apply(lambda x : dfAds[dfAds.idAds==x].inseeCode.values[0])
attributes_rdf = attributes_rdf.drop("idAds",axis=1).drop_duplicates()
attributes_rdf = attributes_rdf[attributes_rdf.attributes_clean.isin(attributes_rdf.attributes_clean.value_counts().loc[lambda x: x>=2].index)]

df_finalKG.to_pickle("./results/dfFinalKG.pkl")
dfFeature.to_pickle("./results/dfFeature.pkl")
attributes_rdf.to_pickle("./results/attributes_rdf.pkl")

new_sa = sa.SpatialApproximation(df_finalKG)

# for inseeCode in dfAds.inseeCode.unique() :
inseeCode = "06088" 
with open('example_parcels.pkl', 'rb') as f:
    parcels = pickle.load(f)
df_final = new_sa.SpatialApproximationMain(df_finalKG,dfAds,parcels,inseeCode)
df_final.to_pickle("./results/geocoding_"+str(inseeCode)+".pkl")



