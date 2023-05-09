
import pandas as pd
import numpy as np
import scipy
from scipy.stats import chi2
from scipy.stats import gaussian_kde
import shapely

class SpatialApproximation():


    def point_outliers(self,df_spatial):
    
        df_numpy = df_spatial[["long_float","lat_float"]].to_numpy() 
        # Covariance matrix
        covariance  = np.cov(df_numpy , rowvar=False)

        # Covariance matrix power of -1
        covariance_pm1 = np.linalg.matrix_power(covariance, -1)

        # Center point
        centerpoint = np.mean(df_numpy , axis=0)
        distances = []
        for i, val in enumerate(df_numpy):
            p1 = val
            p2 = centerpoint
            distance = (p1-p2).T.dot(covariance_pm1).dot(p1-p2)
            distances.append(distance)
        distances = np.array(distances)

        # Cutoff (threshold) value from Chi-Sqaure Distribution for detecting outliers 
        cutoff = chi2.ppf(0.99, df_numpy.shape[1])

        # Index of outliers
        outlierIndexes = np.where(distances > cutoff )
        
        return outlierIndexes
    
    def score_final(self,score): 
        if(score >= 0.9) : 
            return 1
        elif(score >= 0.8):
            return 0.8
        elif(score >= 0.6):
            return 0.6
        elif(score >= 0.4):
            return 0.4
        elif(score >= 0.2):
            return 0.2
        else:
            return 0
    
    def SpatialApproximationMain(self,dfKG,dfAds,parcels,inseeCode):
        
        print(inseeCode)
        df_inseeCode = dfKG[dfKG.idAds.isin(dfAds[dfAds["inseeCode"]==inseeCode].idAds.unique())]
        count = pd.DataFrame(df_inseeCode.et_eg_toponym_new.value_counts())
        place_to_estimate = list(count[count['et_eg_toponym_new']>=10].index)


        lst_to_evaluate = []
        for i in range(0,len(parcels)):
            coord = shapely.wkt.loads(parcels[i][3]).coords[0]
            lst_to_evaluate.append([coord[1],coord[0]])

        df_final = pd.DataFrame(parcels,columns=["uid","source_id","geometry","rep_point"])#.set_index("uid")
        df_final.geometry=df_final.geometry.apply(lambda wkt: shapely.wkt.loads(wkt))

        i=0
        for place in place_to_estimate:
    #         print(i)
            try:
                df =df_inseeCode[df_inseeCode.et_eg_toponym_new==place]
                data =dfAds[dfAds.idAds.isin(df.idAds.unique())]
                data['lat_float'] = data.latitude.apply(lambda x : float(x))
                data['long_float'] = data.longitude.apply(lambda x : float(x))
                index = self.point_outliers(data)
                data_tot = data[~data.index.isin(index[0])]

                kde = gaussian_kde(np.vstack([data_tot.latitude,data_tot.longitude]), bw_method="scott")

                eval_pred = []
                for coord in lst_to_evaluate : 
                    eval_pred.append(kde.pdf(coord))

                df_pred =pd.DataFrame(eval_pred)
                maxi = df_pred.max()
                mini = df_pred.min()
                df_pred["normalized"] = df_pred.apply(lambda x : (x-mini[0])/(maxi[0]-mini[0]))
                df_final["score"] = df_pred["normalized"]
                df_final["LD"+"_"+"_".join(place.split(" "))] = df_final.score.apply(lambda x : self.score_final(x))
            except: 
                print("Not Enough Ads")
            i=i+1
        return df_final
