"""Training-only variance pilot; no validation/test outcomes are used."""
from pathlib import Path
import sys,json,itertools
import numpy as np
import pandas as pd
import scipy,sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
R=Path(__file__).resolve().parent
o=pd.read_csv(R/'processed_orders.csv',dtype={'order_month':str,'order_weekday':str})
t=o.loc[o.partition.eq('train')].copy().reset_index(drop=True)
y=t.Late_delivery_risk.to_numpy(int)
num=['order_quantity','weighted_unit_price','discount_rate_mean','mixed_category']
cat=['Shipping Mode','Market','Customer Segment','Type','order_month','order_weekday','Order Region','dominant_category']
pred=np.zeros((len(t),3));share=np.zeros(len(t))
cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=644)
for fold,(a,b) in enumerate(cv.split(t,y),1):
 print('Training-only pilot fold',fold,flush=True)
 pre=ColumnTransformer([('num',StandardScaler(),num),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat)])
 X=pre.fit_transform(t.iloc[a][num+cat]);V=pre.transform(t.iloc[b][num+cat])
 models=[LogisticRegression(C=1,max_iter=2000,solver='lbfgs',random_state=640),RandomForestClassifier(n_estimators=240,max_depth=14,min_samples_leaf=30,max_features='sqrt',n_jobs=1,random_state=640),HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,learning_rate=.07,l2_regularization=5,min_samples_leaf=40,early_stopping=False,random_state=640)]
 for j,m in enumerate(models):m.fit(X,y[a]);pred[b,j]=m.predict_proba(V)[:,1]
 c=(V-X.mean(axis=0))*models[0].coef_[0];groups=[c[:,j] for j in range(len(num))];offset=len(num)
 for levels in pre.named_transformers_['cat'].categories_:
  groups.append(c[:,offset:offset+len(levels)].sum(axis=1));offset+=len(levels)
 G=np.column_stack(groups);den=abs(G).sum(axis=1);share[b]=np.divide(abs(G[:,len(num)]),den,out=np.zeros(len(b)),where=den>0)
loss=(pred-y[:,None])**2
pairs=[{'pair':f'{a} minus {b}','paired_loss_sd':float((loss[:,i]-loss[:,j]).std(ddof=1))} for (i,a),(j,b) in itertools.combinations(enumerate(['Logistic','Forest','Boosting']),2)]
segments=[]
for c in ['Shipping Mode','Market']:
 for level in sorted(t[c].unique()):
  m=t[c].eq(level).to_numpy();segments.append({'family':c,'segment':level,'n':int(m.sum()),'normalized_share_sd':float(share[m].std(ddof=1))})
r={'training_only_n':len(t),'folds':3,'seed':644,'method':'Stratified out-of-fold predictions and fold-specific training backgrounds; no validation/test outcomes','paired_loss_variability':pairs,'segment_variability':segments,'max_paired_sd':max(x['paired_loss_sd'] for x in pairs),'max_segment_sd':max(x['normalized_share_sd'] for x in segments),'planning_sd_retained':.2,'effect_size_rq3':.01,'effect_size_rq4':.05,'effect_size_rationale':'RQ3 .01 is an absolute probability-loss planning threshold (about 4% of a .25 benchmark); RQ4 .05 is a five-percentage-point attribution-share contrast. Neither is a validated business value or an observed effect.'}
(R/'pilot_planning_results.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
pd.DataFrame(pairs).to_csv(R/'pilot_paired_variability.csv',index=False);pd.DataFrame(segments).to_csv(R/'pilot_segment_variability.csv',index=False)
print(json.dumps(r,indent=2),flush=True)
