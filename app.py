from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
import json

app = Flask(__name__)

def init_models():
    chemin_fichier = "ProjetM2SID2026.xlsx"
    df = pd.read_excel(chemin_fichier, sheet_name="Donnees")
    
    # 1. Nettoyage des noms de colonnes
    df.columns = df.columns.str.strip()
    
    time_col = "DUREE SUIVI Apres Traitement (mois)"
    event_col = "DECES_NUM"
    
    # 2. Gestion des valeurs manquantes et encodages
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce').fillna(df[time_col].median())
    df[event_col] = df['DECES'].map({'OUI': 1, 'NON': 0}).fillna(0).astype(int)
    
    def map_column(search_term, new_name, default_val=0):
        actual_col = [c for c in df.columns if search_term.lower() in c.lower()]
        if actual_col:
            df[new_name] = df[actual_col[0]].map({'OUI': 1, 'NON': 0, 'M': 1, 'F': 0})
            df[new_name] = df[new_name].fillna(df[actual_col[0]]).fillna(default_val).astype(int)
        else:
            df[new_name] = default_val

    map_column('DIABETE', 'DIABETE_NUM', default_val=0)
    map_column('Metastases Hepatiques', 'Metastases Hepatiques_NUM', default_val=0)
    map_column('Dénutrition', 'Dénutrition_NUM', default_val=0)
    map_column('SEXE', 'SEXE_NUM', default_val=1)
    map_column('chirurgie', 'Traitement par chirurgie_NUM', default_val=1)
    
    df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce').fillna(df['AGE'].mean())
    
    col_hemo = [c for c in df.columns if 'hémoglobine' in c.lower() or 'hemo' in c.lower()]
    df['hémoglobine'] = pd.to_numeric(df[col_hemo[0]], errors='coerce').fillna(df['hémoglobine'].mean()) if col_hemo else 12.0
    
    col_sympt = [c for c in df.columns if 'sympt' in c.lower() or 'evolution' in c.lower()]
    df["Durée d'evolution des Symptom en Mois"] = pd.to_numeric(df[col_sympt[0]], errors='coerce').fillna(6.0) if col_sympt else 6.0

    # Caractéristiques partagées
    features_cox = ['AGE', 'SEXE_NUM', 'hémoglobine', "Durée d'evolution des Symptom en Mois", 
                    'DIABETE_NUM', 'Metastases Hepatiques_NUM', 'Dénutrition_NUM', 'Traitement par chirurgie_NUM']
    
    # --- ENTRAÎNEMENT MODÈLE 1 : COX ---
    df_cox = df[features_cox + [time_col, event_col]].copy()
    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col=time_col, event_col=event_col)
    
    # --- ENTRAÎNEMENT MODÈLE 2 : RÉGRESSION LOGISTIQUE ---
    features_ml = ['AGE', 'SEXE_NUM', 'hémoglobine', 'DIABETE_NUM', 'Metastases Hepatiques_NUM', 'Dénutrition_NUM', 'Traitement par chirurgie_NUM']
    X_ml = df[features_ml].copy()
    y_ml = df[event_col].copy()
    
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_ml, y_ml)
    
    return cph, log_reg, features_ml

# Initialisation des deux modèles
model_cox, model_lr, features_ml = init_models()

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_data = None
    score_risque = None
    probabilite_ml = None
    
    if request.method == 'POST':
        age = float(request.form.get('age', 65))
        sexe = int(request.form.get('sexe', 1))
        hemo = float(request.form.get('hemo', 11.0))
        sympt = float(request.form.get('sympt', 6))
        diabete = int(request.form.get('diabete', 0))
        metastase = int(request.form.get('metastase', 0))
        denutrition = int(request.form.get('denutrition', 0))
        chirurgie = int(request.form.get('chirurgie', 1))
        
        # Profil complet pour Cox
        profil_cox = pd.DataFrame([{
            'AGE': age, 'SEXE_NUM': sexe, 'hémoglobine': hemo,
            "Durée d'evolution des Symptom en Mois": sympt, 'DIABETE_NUM': diabete,
            'Metastases Hepatiques_NUM': metastase, 'Dénutrition_NUM': denutrition,
            'Traitement par chirurgie_NUM': chirurgie
        }])
        
        # 1. Calculs Cox
        surv_prob = model_cox.predict_survival_function(profil_cox)
        prediction_data = {
            "labels": list(surv_prob.index.astype(int)),
            "values": list(np.round(surv_prob.values.flatten(), 2))
        }
        score_risque = round(float(model_cox.predict_partial_hazard(profil_cox).values[0]), 4)
        
        # 2. Calculs Régression Logistique (Uniquement sur les colonnes correspondantes)
        profil_ml = profil_cox[features_ml]
        probabilite_ml = round(float(model_lr.predict_proba(profil_ml)[0][1]) * 100, 2)

    return render_template('index.html', 
                           prediction_data=json.dumps(prediction_data), 
                           score_risque=score_risque, 
                           probabilite_ml=probabilite_ml)

if __name__ == '__main__':
    app.run(debug=True)