"""Validasi diagnostik pola target dan kestabilan model pada dataset Kaggle."""
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'diabetes_dataset00.csv'
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
CLASSES=['Prediabetic','Type 1 Diabetes','Type 2 Diabetes']
SEED=42

def main():
    raw=pd.read_csv(DATA); df=raw[raw.Target.isin(CLASSES)].copy()
    X=df.drop(columns='Target'); y=df.Target.astype(str)
    # Rule diagnostik yang diperoleh dari inspeksi subset; bukan model klinis.
    rule=[]
    for _,r in X.iterrows():
        if r['Age'] < 30: rule.append('Type 1 Diabetes')
        elif r['Blood Glucose Levels'] < 125: rule.append('Prediabetic')
        else: rule.append('Type 2 Diabetes')
    rule_acc=accuracy_score(y,rule)
    (OUT/'rule_diagnostic.json').write_text(json.dumps({
        'accuracy_on_full_selected_subset':float(rule_acc),
        'warning':'Aturan diperiksa setelah melihat keseluruhan subset; bukan evaluasi independen dan tidak boleh dipakai untuk diagnosis.'
    },indent=2),encoding='utf-8')

    num=X.select_dtypes(include='number').columns.tolist(); cat=X.select_dtypes(exclude='number').columns.tolist()
    pre=ColumnTransformer([
        ('num',SimpleImputer(strategy='median'),num),
        ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)
    ])
    models={
      'RandomForest':RandomForestClassifier(n_estimators=200,random_state=SEED,n_jobs=1),
      'GradientBoosting':GradientBoostingClassifier(n_estimators=100,learning_rate=.08,max_depth=3,random_state=SEED),
      'LightGBM':LGBMClassifier(n_estimators=200,learning_rate=.05,num_leaves=31,verbosity=-1,n_jobs=1,random_state=SEED)
    }
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    rows=[]
    for name,est in models.items():
      pipe=Pipeline([('preprocess',pre),('model',est)])
      print('CV:',name,flush=True)
      scores=cross_validate(pipe,X,y,cv=cv,scoring={'accuracy':'accuracy','f1_macro':'f1_macro'},n_jobs=1,return_train_score=False)
      rows.append({'model':name,'accuracy_mean':scores['test_accuracy'].mean(),'accuracy_std':scores['test_accuracy'].std(),'f1_macro_mean':scores['test_f1_macro'].mean(),'f1_macro_std':scores['test_f1_macro'].std()})
    pd.DataFrame(rows).to_csv(OUT/'five_fold_cv_diagnostic.csv',index=False)
    print('Rule accuracy (diagnostic only):',rule_acc)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__': main()
