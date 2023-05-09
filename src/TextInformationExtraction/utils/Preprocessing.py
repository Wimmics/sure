import re
import pandas as pd
import unidecode

class Preprocessing() : 

    def clean_text(your_text):
    
        """
            Fonction qui nettoie les textes (url,email,...,espaces,etc.)
        """
    
        texte = re.sub(r'(http|www)\S+',r"",your_text) #enlever les url
        texte = re.sub(r"Obtenir l'adresse","." ,texte, flags=re.IGNORECASE)
        texte = re.sub(r"Être rappelé","." ,texte, flags=re.IGNORECASE)
        texte = re.sub(r"Demander une visite","." ,texte, flags=re.IGNORECASE)
        texte = re.sub(r'm(²|2)',' m² ',texte, flags=re.IGNORECASE)
        texte = re.sub(r'euro[s]|€|\beur\b',' euro ',texte, flags=re.IGNORECASE)
        texte = re.sub(r'(excl)(\d*)(/)(\d*)j',' ',texte, flags=re.IGNORECASE)
        texte = re.sub(r'(\+33 |0+)[1-9](\s*\d\d){4}' ," ",texte)
        texte = re.sub(r'(\bt(e|è)l\b)\s+' ," ",texte, flags=re.IGNORECASE)
        texte = re.sub(r'(r[e|é|è]f)(\s*\[.]*\s*)((annonce)\s*\[:]*\s*(\d*)|\d*)',' ',texte, flags=re.IGNORECASE)
        
        texte = re.sub(r'\r+\n+',' ',texte)
        texte = re.sub(r"([.]\s+[.]+)",r". ",texte) #recoller les  "... . ." en "....."
        texte = re.sub(r"([,]\s+[,]+)",r", ",texte) #recoller les  ", , ," en ",,,"
        texte = re.sub(r"[(]\s*[)]",r"",texte) # supprimer ()
        texte = re.sub(r'([^A-Za-z0-9])(\1)+',r"\1 ",texte)
        texte = re.sub(r'(?<=[.,])(?=[^\s])', r'', texte)
        texte = re.sub(r'/', r' / ', texte)
        texte = re.sub(r'\\+', r' ', texte)
        
        texte = re.sub(r"(\d+)\s+(\d+)",r'\1\2',texte) #concatener les nombres avec un espace (3 600 € -> 3600 €)
        texte = re.sub(r"(\d+)([Aa-zZ]+)",r'\1 \2',texte,flags=re.IGNORECASE) #4pièces -> 4 pièces/ 10m² -> 10 m²
        texte = re.sub(r"(.)([,.])(.)",r'\1 \2 \3',texte,flags=re.IGNORECASE)
        texte = re.sub(r"(\d+)\s*([,.])\s*(\d+)",r'\1\2\3',texte) # 54 , 5 m² -> 54,5 m² 
        texte = re.sub(r"('|-|<|>|`|#|_|=|&|~|^|¨|’|–|…)+", r" ", texte)
        texte = texte.replace("*", " ")
        texte = texte.replace("+"," ")
        texte = re.sub(r'\s+eur\s',' euro ',texte, flags=re.IGNORECASE)

        
        texte = re.sub(r'\s+',' ',texte)
        
        texte = texte.lstrip(" . ")
        texte = texte.lstrip()
        
        
    
        return texte
