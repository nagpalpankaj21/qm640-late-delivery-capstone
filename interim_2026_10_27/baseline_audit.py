from pathlib import Path
import hashlib,json,math,argparse
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description='Reproduce the QM640 interim analysis.')
parser.add_argument('--data',type=Path,default=ROOT/'DataCoSupplyChainDataset.csv')
SOURCE=parser.parse_args().data
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='fa6d022ed437155e1a2f0378710602848703c8a7f203f7ff5d77805bf8480aa6','Source checksum mismatch'
df=pd.read_csv(SOURCE,encoding='latin-1')
meta=['Late_delivery_risk','Shipping Mode','Market','Order Region','Customer Segment','Type','Days for shipment (scheduled)','order date (DateOrders)']
g=df.groupby('Order Id',sort=True)
conf={c:int((g[c].nunique(dropna=False)>1).sum()) for c in meta}
badids=set()
for c in meta: badids.update(g[c].nunique(dropna=False).loc[lambda x:x>1].index)
q=pd.to_numeric(df['Order Item Quantity'],errors='coerce')
price=pd.to_numeric(df['Order Item Product Price'],errors='coerce')
disc=pd.to_numeric(df['Order Item Discount Rate'],errors='coerce')
invalid=(~np.isfinite(q))|(q<=0)|(~np.isfinite(price))|(price<=0)|(~np.isfinite(disc))|(disc<0)|(disc>1)
badids.update(df.loc[invalid,'Order Id'])
dt=pd.to_datetime(df['order date (DateOrders)'],format='%m/%d/%Y %H:%M',errors='coerce')
badids.update(df.loc[dt.isna(),'Order Id'])
df['_wp']=q*price;df['_wd']=q*disc
o=g[meta].first()
o['order_quantity']=g['Order Item Quantity'].sum()
o['weighted_unit_price']=g['_wp'].sum()/o.order_quantity
o['discount_rate_mean']=g['_wd'].sum()/o.order_quantity
o['line_count']=g.size()
for raw,new in [('Category Name','dominant_category'),('Department Name','dominant_department')]:
 a=df.groupby(['Order Id',raw])['Order Item Quantity'].sum().reset_index().sort_values(['Order Id','Order Item Quantity',raw],ascending=[True,False,True])
 o[new]=a.drop_duplicates('Order Id').set_index('Order Id')[raw]
o['mixed_category']=(g['Category Name'].nunique()>1).astype(int)
o['order_date']=pd.to_datetime(o['order date (DateOrders)'],format='%m/%d/%Y %H:%M',errors='coerce')
o['order_month']=o.order_date.dt.month.astype(str);o['order_weekday']=o.order_date.dt.dayofweek.astype(str)
o=o.loc[~o.index.isin(badids)].copy()
y=o.Late_delivery_risk.to_numpy(float)
# Freeze a stratified 60/20/20 order split before analytical displays.
rng=np.random.default_rng(640)
split=np.full(len(o),'',dtype='<U10')
for label in [0,1]:
 ix=np.flatnonzero(y==label);rng.shuffle(ix);a=int(.6*len(ix));b=int(.8*len(ix))
 split[ix[:a]]='train';split[ix[a:b]]='validation';split[ix[b:]]='test'
o['partition']=split
tr=np.flatnonzero(split=='train');va=np.flatnonzero(split=='validation')
dev=o.loc[o.partition!='test']
def group_summary(data,col):
 a=data.groupby(col).Late_delivery_risk.agg(['size','sum','mean']).reset_index()
 return json.loads(a.to_json(orient='records'))
numeric=['order_quantity','weighted_unit_price','discount_rate_mean']
categorical=['Shipping Mode','Market','Customer Segment','Type','order_month','order_weekday']
# Restricted, predeclared baseline avoids redundant shipping-duration and nested geography.
means=o.iloc[tr][numeric].mean();sd=o.iloc[tr][numeric].std(ddof=0)
Xparts=[np.ones((len(o),1)),((o[numeric]-means)/sd).to_numpy()]
names=['intercept']+numeric
levels={}
for c in categorical:
 levels[c]=sorted(o.iloc[tr][c].unique().tolist())
 for lev in levels[c][1:]:
  Xparts.append((o[c].to_numpy()==lev).astype(float)[:,None]);names.append(c+' = '+lev)
X=np.column_stack(Xparts)
beta=np.zeros(X.shape[1]);penalty=np.ones(X.shape[1]);penalty[0]=0
# L2 penalty lambda=1, objective sum binary log loss + .5*||beta||^2.
for step in range(60):
 z=np.clip(X[tr]@beta,-30,30);p=1/(1+np.exp(-z));w=p*(1-p)
 grad=X[tr].T@(p-y[tr])+penalty*beta
 H=(X[tr].T*w)@X[tr]+np.diag(penalty+1e-10)
 delta=np.linalg.solve(H,grad);beta-=delta
 if max(abs(delta))<1e-8: break
pv=1/(1+np.exp(-np.clip(X[va]@beta,-30,30)))
pt=float(y[tr].mean())
mode_rates=o.iloc[tr].groupby('Shipping Mode').Late_delivery_risk.mean()
pm=o.iloc[va]['Shipping Mode'].map(mode_rates).to_numpy()
def metrics(yy,pp):
 pred=pp>=.5;tp=int(((yy==1)&pred).sum());fp=int(((yy==0)&pred).sum());tn=int(((yy==0)&~pred).sum());fn=int(((yy==1)&~pred).sum())
 ranks=pd.Series(pp).rank(method='average').to_numpy();n1=(yy==1).sum();n0=(yy==0).sum()
 auc=float((ranks[yy==1].sum()-n1*(n1+1)/2)/(n1*n0))
 # Average precision grouped by distinct score, including tied probabilities.
 ag=pd.DataFrame({'p':pp,'y':yy}).groupby('p').y.agg(['sum','count']).sort_index(ascending=False)
 ap=float(((ag['sum'].cumsum()/ag['count'].cumsum())*ag['sum']/n1).sum())
 return dict(n=len(yy),tp=tp,fp=fp,tn=tn,fn=fn,accuracy=(tp+tn)/len(yy),precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn),specificity=tn/(tn+fp),f1=2*tp/(2*tp+fp+fn),auc=auc,average_precision=ap,brier=float(np.mean((pp-yy)**2)),alert_rate=float(pred.mean()))
preds={'Constant probability':np.full(len(va),pt),'Shipping mode lookup':pm,'Logistic regression':pv}
result={k:metrics(y[va],v) for k,v in preds.items()}
boot=[];brng=np.random.default_rng(641)
for _ in range(500):
 ix=brng.integers(0,len(va),len(va)); yy=y[va][ix];pp=pv[ix]
 ranks=pd.Series(pp).rank(method='average').to_numpy();n1=(yy==1).sum();n0=(yy==0).sum()
 boot.append([(ranks[yy==1].sum()-n1*(n1+1)/2)/(n1*n0),np.mean((pp-yy)**2),np.mean((pm[ix]-yy)**2)-np.mean((pp-yy)**2)])
ci=np.quantile(boot,[.025,.975],axis=0).T.tolist()
cal=[]
for a in np.linspace(0,.9,10):
 mask=(pv>=a)&(pv<(a+.1) if a<.9 else pv<=1)
 if mask.any(): cal.append(dict(lower=round(float(a),1),n=int(mask.sum()),predicted=float(pv[mask].mean()),observed=float(y[va][mask].mean())))
ece=sum(c['n']/len(va)*abs(c['predicted']-c['observed']) for c in cal)
outliers={}
for c in numeric:
 v=o.iloc[tr][c];q1,q3=v.quantile([.25,.75]);lo=q1-1.5*(q3-q1);hi=q3+1.5*(q3-q1)
 outliers[c]=dict(lower=float(lo),upper=float(hi),flagged_development=int(((dev[c]<lo)|(dev[c]>hi)).sum()))
year=dev.assign(year=dev.order_date.dt.year)
summ={
 'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'raw_rows':len(df),'raw_columns':53,'orders':len(o),'excluded_orders':len(badids),'conflicts':conf,
 'duplicates_full':int(df.drop(columns=['_wp','_wd']).duplicated().sum()),'duplicate_line_ids':int(df['Order Item Id'].duplicated().sum()),'invalid_lines':int(invalid.sum()),'date_parse_failures':int(dt.isna().sum()),
 'raw_missing':{c:int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()},
 'splits':{k:dict(n=int((split==k).sum()),positive=int(y[split==k].sum())) for k in ['train','validation','test']},
 'development_n':len(dev),'development_prevalence':float(dev.Late_delivery_risk.mean()),
 'by_mode':group_summary(dev,'Shipping Mode'),'by_market':group_summary(dev,'Market'),'by_scheduled':group_summary(dev,'Days for shipment (scheduled)'),
 'by_year':group_summary(year,'year'),'by_segment':group_summary(dev,'Customer Segment'),
 'numeric_summary':json.loads(dev[numeric].describe().round(4).to_json()),'outliers':outliers,
 'mixed_category_development':int(dev.mixed_category.sum()),'metrics':result,'ci':ci,'calibration':cal,'ece':ece,'iterations':step+1,
 'coefficients':dict(zip(names,beta.tolist())),'features':numeric+categorical,'training_prevalence':pt,
 'date_min':str(o.order_date.min()),'date_max':str(o.order_date.max()),'software':{'python':__import__('platform').python_version(),'numpy':np.__version__,'pandas':pd.__version__},
}
(ROOT/'analysis_results.json').write_text(json.dumps(summ,indent=2))
# Identifiers never exported. Feature records carry no original order/customer key.
exportcols=meta[:7]+numeric+['dominant_category','dominant_department','mixed_category','order_month','order_weekday','partition']
o[exportcols].to_csv(ROOT/'processed_orders.csv',index=False)
pd.DataFrame({'y':y[va],'constant_probability':pt,'mode_probability':pm,'logistic_probability':pv}).to_csv(ROOT/'validation_predictions.csv',index=False)
pd.DataFrame({'feature':names,'coefficient':beta}).to_csv(ROOT/'logistic_coefficients.csv',index=False)
print(json.dumps(summ,indent=2))
