from pathlib import Path
import sys, json, math, itertools, platform
import numpy as np
import pandas as pd
import scipy,sklearn
from scipy.special import expit
from scipy.optimize import minimize
from scipy.stats import norm, chi2
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix, log_loss
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)

R=Path(__file__).resolve().parent
o=pd.read_csv(R/'processed_orders.csv',dtype={'order_month':str,'order_weekday':str})
rawpath=Path(sys.argv[1])
raw=pd.read_csv(rawpath,encoding='latin-1',usecols=['Order Id','Customer Id','order date (DateOrders)','Delivery Status'])
keys=raw.groupby('Order Id',sort=True).first()
assert len(keys)==len(o)
o['_date']=pd.to_datetime(keys['order date (DateOrders)'].to_numpy(),format='%m/%d/%Y %H:%M')
o['_customer']=keys['Customer Id'].to_numpy()
o['_completed']=keys['Delivery Status'].to_numpy()!='Shipping canceled'
train=o.partition.eq('train').to_numpy();val=o.partition.eq('validation').to_numpy();dev=train|val
y=o.Late_delivery_risk.to_numpy(int)
num=['order_quantity','weighted_unit_price','discount_rate_mean','mixed_category']
cat=['Shipping Mode','Market','Customer Segment','Type','order_month','order_weekday','Order Region','dominant_category']
features=num+cat
pre=ColumnTransformer([('num',StandardScaler(),num),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat)])
Xtr=pre.fit_transform(o.loc[train,features]);Xv=pre.transform(o.loc[val,features])
models={
 'Logistic full':LogisticRegression(C=1,max_iter=2000,solver='lbfgs',random_state=640),
 'Random forest':RandomForestClassifier(n_estimators=240,max_depth=14,min_samples_leaf=30,max_features='sqrt',n_jobs=1,random_state=640),
 'Gradient boosting':HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,learning_rate=.07,l2_regularization=5,min_samples_leaf=40,early_stopping=False,random_state=640)
}
def metrics(yy,p):
 tn,fp,fn,tp=confusion_matrix(yy,p>=.5,labels=[0,1]).ravel()
 return dict(n=len(yy),auc=roc_auc_score(yy,p),ap=average_precision_score(yy,p),brier=brier_score_loss(yy,p),logloss=log_loss(yy,p),tp=int(tp),tn=int(tn),fp=int(fp),fn=int(fn),accuracy=(tp+tn)/len(yy),precision=tp/(tp+fp),recall=tp/(tp+fn),specificity=tn/(tn+fp),f1=2*tp/(2*tp+fp+fn),alerts=(tp+fp)/len(yy))
old=pd.read_csv(R/'validation_predictions.csv')
assert np.array_equal(old.y,y[val])
pred={ 'Constant':old.constant_probability.to_numpy(), 'Mode lookup':old.mode_probability.to_numpy(), 'Logistic restricted':old.logistic_probability.to_numpy() }
for name,m in models.items():
 print('Fitting',name,flush=True);m.fit(Xtr,y[train]);pred[name]=m.predict_proba(Xv)[:,1]
out={'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__},'feature_count_encoded':int(Xtr.shape[1]),'model_settings':{n:m.get_params() for n,m in models.items()},'metrics':{n:metrics(y[val],p) for n,p in pred.items()}}
# Frozen validation comparisons are exploratory; all three primary contrasts use common orders.
brng=np.random.default_rng(642);primary=list(models);boot={n:[] for n in pred};diffs={a+' vs '+b:[] for a,b in itertools.combinations(primary,2)}
for _ in range(1000):
 ix=brng.integers(0,val.sum(),val.sum());yy=y[val][ix]
 for n,p in pred.items(): boot[n].append([roc_auc_score(yy,p[ix]),np.mean((p[ix]-yy)**2)])
 for a,b in itertools.combinations(primary,2): diffs[a+' vs '+b].append(float(np.mean((pred[a][ix]-yy)**2-(pred[b][ix]-yy)**2)))
out['intervals']={n:np.quantile(v,[.025,.975],axis=0).T.tolist() for n,v in boot.items()}
out['paired_brier']={a+' vs '+bb:{'difference':float(np.mean((pred[a]-y[val])**2-(pred[bb]-y[val])**2)),'ci9833':np.quantile(diffs[a+' vs '+bb],[.05/6,1-.05/6]).tolist()} for a,bb in itertools.combinations(primary,2)}
pd.DataFrame({'y':y[val],**pred}).to_csv(R/'enhanced_validation_predictions.csv',index=False)
# RQ1: prespecified Standard Class contrast, no redundant scheduled duration or nested geography.
infer=o.loc[dev].copy();infer['standard_class']=infer['Shipping Mode'].eq('Standard Class').astype(int)
inum=['standard_class','order_quantity','weighted_unit_price','discount_rate_mean']
icat=['Market','Customer Segment','Type','order_month','order_weekday']
ipre=ColumnTransformer([('num','passthrough',inum),('cat',OneHotEncoder(drop='first',sparse_output=False),icat)])
IX=np.column_stack([np.ones(len(infer)),ipre.fit_transform(infer[inum+icat])]);iy=infer.Late_delivery_risk.to_numpy()
names=['intercept']+inum+list(ipre.named_transformers_['cat'].get_feature_names_out(icat))
# Improve numerical conditioning while preserving interpretable increments.
scales=np.ones(IX.shape[1]);scales[2]=5;scales[3]=100;scales[4]=.1
IX=IX/scales
fit=minimize(lambda b:np.logaddexp(0,IX@b).sum()-iy@(IX@b),np.zeros(IX.shape[1]),jac=lambda b:IX.T@(expit(IX@b)-iy),method='BFGS',options={'gtol':1e-6,'maxiter':500})
b=fit.x;pp=expit(IX@b);H=(IX.T*(pp*(1-pp)))@IX;bread=np.linalg.inv(H)
scores=IX*(iy-pp)[:,None];g=infer['_customer'].to_numpy();unique,inv=np.unique(g,return_inverse=True);sg=np.zeros((len(unique),IX.shape[1]));np.add.at(sg,inv,scores)
cov=bread@(sg.T@sg)@bread*len(unique)/(len(unique)-1)*(len(infer)-1)/(len(infer)-IX.shape[1]);se=np.sqrt(np.diag(cov))
irt=pd.DataFrame({'feature':names,'coefficient':b,'se_customer_clustered':se,'odds_ratio':np.exp(b),'ci_low':np.exp(b-1.96*se),'ci_high':np.exp(b+1.96*se),'p':2*norm.sf(abs(b/se))})
irt.to_csv(R/'adjusted_associations.csv',index=False)
wald=float(b[1:]@np.linalg.solve(cov[1:,1:],b[1:]))
out['inference']={'n':len(infer),'customer_clusters':len(unique),'max_gradient':float(np.max(abs(IX.T@(pp-iy)))),'hessian_condition':float(np.linalg.cond(H)),'coefficient_count':len(b),'wald_all':wald,'wald_df':len(b)-1,'wald_p':float(chi2.sf(wald,len(b)-1)),'rows':irt.iloc[1:5].to_dict('records')}
# Exact additive interventional explanations on logistic log odds, grouped by source variable.
lm=models['Logistic full'];beta=lm.coef_[0];contrib=(Xv-Xtr.mean(axis=0))*beta
groups={};offset=0
for j,n in enumerate(num):groups[n]=contrib[:,j];offset+=1
for n,levels in zip(cat,pre.named_transformers_['cat'].categories_):
 groups[n]=contrib[:,offset:offset+len(levels)].sum(axis=1);offset+=len(levels)
assert offset==Xv.shape[1]
G=np.column_stack(list(groups.values()));gn=list(groups)
assert np.max(abs(lm.intercept_[0]+Xtr.mean(axis=0)@beta+G.sum(axis=1)-lm.decision_function(Xv)))<1e-10
globalimp=pd.DataFrame({'feature':gn,'mean_absolute_logodds':np.abs(G).mean(axis=0)}).sort_values('mean_absolute_logodds',ascending=False)
globalimp.to_csv(R/'logistic_grouped_explanations.csv',index=False)
principal='Shipping Mode' # predeclared from earlier training/development evidence, never validation ranking selected
S=np.divide(abs(G[:,gn.index(principal)]),abs(G).sum(axis=1),out=np.zeros(len(G)),where=abs(G).sum(axis=1)>0)
seg=[]
for c in ['Shipping Mode','Market']:
 for level in sorted(o.loc[val,c].unique()):
  mask=o.loc[val,c].eq(level).to_numpy();a=S[mask];sem=a.std(ddof=1)/math.sqrt(len(a))
  seg.append({'segment_type':c,'segment':level,'n':len(a),'mean_normalized_mode':float(a.mean()),'ci_low':float(a.mean()-1.96*sem),'ci_high':float(a.mean()+1.96*sem),**metrics(y[val][mask],pred['Logistic full'][mask])})
out['segments']=seg;out['explanations']=globalimp.to_dict('records')
pd.DataFrame(seg).to_csv(R/'segment_results.csv',index=False)
contr=[]
for c in ['Shipping Mode','Market']:
 levels=[s for s in seg if s['segment_type']==c];k=math.comb(len(levels),2);z=norm.ppf(1-.05/(2*k))
 for a,bb in itertools.combinations(levels,2):
  aa=S[o.loc[val,c].eq(a['segment']).to_numpy()];ab=S[o.loc[val,c].eq(bb['segment']).to_numpy()]
  d=aa.mean()-ab.mean();seD=math.sqrt(aa.var(ddof=1)/len(aa)+ab.var(ddof=1)/len(ab))
  contr.append({'family':c,'a':a['segment'],'b':bb['segment'],'difference':float(d),'low':float(d-z*seD),'high':float(d+z*seD)})
out['segment_contrasts']=contr;pd.DataFrame(contr).to_csv(R/'segment_contrasts.csv',index=False)
# Controlled validation feature removal: retrain same logistic recipe without shipping mode.
ablcat=[x for x in cat if x!='Shipping Mode'];apre=ColumnTransformer([('num',StandardScaler(),num),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),ablcat)])
ab=make_pipeline(apre,LogisticRegression(C=1,max_iter=2000,random_state=640));ab.fit(o.loc[train,num+ablcat],y[train]);out['without_mode']=metrics(y[val],ab.predict_proba(o.loc[val,num+ablcat])[:,1])
# Temporal sensitivity uses development orders only; never touches reserved test predictions.
early=dev & (o['_date']<pd.Timestamp('2017-01-01'));late=dev & (o['_date']>=pd.Timestamp('2017-01-01'))
tp=ColumnTransformer([('num',StandardScaler(),num),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat)])
tm=make_pipeline(tp,LogisticRegression(C=1,max_iter=2000,random_state=640));tm.fit(o.loc[early,features],y[early]);out['temporal']={'train_n':int(early.sum()),'late_start':'2017-01-01',**metrics(y[late],tm.predict_proba(o.loc[late,features])[:,1])}
completed=o.loc[val,'_completed'].to_numpy();out['completed_sensitivity']={'excluded_canceled':int((~completed).sum()),**metrics(y[val][completed],pred['Logistic full'][completed])}
# Training-only, raw-feature permutation importance; validation outcomes do not select model features.
# Explanatory ranking is shown for logistic, not assumed to explain the nonlinear candidate.
z3=norm.ppf(1-.05/6);z4=norm.ppf(1-.05/20);zp=norm.ppf(.8)
n3=math.ceil(((z3+zp)*.2/.01)**2);n4=math.ceil(2*((z4+zp)*.2/.05)**2)
out['sample_size']={'z_rq3':float(z3),'z_rq4':float(z4),'z_power':float(zp),'rq3_test':n3,'rq3_total':math.ceil(n3/.2),'rq4_per_segment':n4,'rq4_total':math.ceil(n4/(.05*.2)),'rq3_sigma03_total':math.ceil(math.ceil(((z3+zp)*.3/.01)**2)/.2),'rq4_sigma03_total':math.ceil(math.ceil(2*((z4+zp)*.3/.05)**2)/.01),'final_n':max(2562,5770,math.ceil(n3/.2),math.ceil(n4/.01))}
out['validation_brier_difference_sd']={a+' vs '+bb:float(np.std((pred[a]-y[val])**2-(pred[bb]-y[val])**2,ddof=1)) for a,bb in itertools.combinations(primary,2)}
out['explanation_sd_by_segment']={s['segment']:float(S[o.loc[val,s['segment_type']].eq(s['segment']).to_numpy()].std(ddof=1)) for s in seg}
# Customer cluster bootstrap recognizes repeated customers across validation orders.
vc=o.loc[val,'_customer'].to_numpy();cl,ci=np.unique(vc,return_inverse=True);crng=np.random.default_rng(643)
cb={n:[] for n in primary};cd={a+' vs '+bb:[] for a,bb in itertools.combinations(primary,2)}
for _ in range(500):
 counts=np.bincount(crng.integers(0,len(cl),len(cl)),minlength=len(cl));w=counts[ci]
 for n in primary:cb[n].append([roc_auc_score(y[val],pred[n],sample_weight=w),np.average((pred[n]-y[val])**2,weights=w)])
 for a,bb in itertools.combinations(primary,2):cd[a+' vs '+bb].append(np.average((pred[a]-y[val])**2-(pred[bb]-y[val])**2,weights=w))
out['customer_bootstrap']={'clusters':len(cl),'replicates':500,'seed':643,'intervals':{n:np.quantile(v,[.025,.975],axis=0).T.tolist() for n,v in cb.items()},'paired9833':{n:np.quantile(v,[.05/6,1-.05/6]).tolist() for n,v in cd.items()}}
# Customer-cluster robust segment mean uncertainty and contrasts.
for s in seg:
 mask=o.loc[val,s['segment_type']].eq(s['segment']).to_numpy();scores=np.zeros(len(S));scores[mask]=(S[mask]-S[mask].mean())/mask.sum()
 clscore=np.bincount(ci,weights=scores,minlength=len(cl));semean=np.sqrt((clscore@clscore)*len(cl)/(len(cl)-1));s['ci_low']=s['mean_normalized_mode']-1.96*semean;s['ci_high']=s['mean_normalized_mode']+1.96*semean
for s in contr:
 ma=o.loc[val,s['family']].eq(s['a']).to_numpy();mb=o.loc[val,s['family']].eq(s['b']).to_numpy();scores=np.zeros(len(S));scores[ma]=(S[ma]-S[ma].mean())/ma.sum();scores[mb]-=(S[mb]-S[mb].mean())/mb.sum()
 clscore=np.bincount(ci,weights=scores,minlength=len(cl));sed=np.sqrt((clscore@clscore)*len(cl)/(len(cl)-1));pairs=6 if s['family']=='Shipping Mode' else 10;z=norm.ppf(1-.05/(2*pairs));s['low']=s['difference']-z*sed;s['high']=s['difference']+z*sed
pd.DataFrame(seg).to_csv(R/'segment_results.csv',index=False);pd.DataFrame(contr).to_csv(R/'segment_contrasts.csv',index=False)
out['calibration']=[]
for low in np.arange(0,1,.1):
 mask=(pred['Logistic full']>=low)&(pred['Logistic full']<(low+.1) if low<.85 else pred['Logistic full']<=1)
 if mask.any():out['calibration'].append({'lower':float(low),'n':int(mask.sum()),'predicted':float(pred['Logistic full'][mask].mean()),'observed':float(y[val][mask].mean())})
out['ece']=sum(a['n']/val.sum()*abs(a['predicted']-a['observed']) for a in out['calibration'])
out['review_budgets']=[]
for fraction in [.1,.2,.35,.5]:
 k=math.ceil(fraction*val.sum());ix=np.argsort(-pred['Logistic full'],kind='stable')[:k];tp=int(y[val][ix].sum());out['review_budgets'].append({'fraction':fraction,'alerts':k,'precision':tp/k,'recall':tp/int(y[val].sum()),'cutoff':float(pred['Logistic full'][ix[-1]])})
(R/'enhanced_results.json').write_text(json.dumps(out,indent=2,default=lambda x:float(x)),encoding='utf-8')
print(json.dumps({k:out[k] for k in ['metrics','inference','without_mode','temporal','completed_sensitivity','sample_size']},indent=2),flush=True)
