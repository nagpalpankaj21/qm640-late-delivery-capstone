"""Development-only target reconciliation and 5,000-draw paired Brier bootstrap."""
from pathlib import Path
import sys,json,itertools,hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent
raw=Path(sys.argv[1])
assert hashlib.sha256(raw.read_bytes()).hexdigest()=='fa6d022ed437155e1a2f0378710602848703c8a7f203f7ff5d77805bf8480aa6'
cols=['Order Id','Customer Id','Days for shipping (real)','Days for shipment (scheduled)','Delivery Status','Late_delivery_risk']
d=pd.read_csv(raw,encoding='latin-1',usecols=cols);g=d.groupby('Order Id',sort=True)
conflicts={c:int((g[c].nunique(dropna=False)>1).sum()) for c in cols[1:]}
assert not any(conflicts.values()),conflicts
o=g.first().reset_index(drop=True);features=pd.read_csv(R/'processed_orders.csv')
assert len(o)==len(features)
assert np.array_equal(o.Late_delivery_risk,features.Late_delivery_risk)
dev=features.partition.ne('test').to_numpy();v=features.partition.eq('validation').to_numpy()
a=o.loc[dev].copy();a['duration_exceeds_promise']=a['Days for shipping (real)']>a['Days for shipment (scheduled)'];a['canceled']=a['Delivery Status'].eq('Shipping canceled')
a['simple_rule_agrees']=a.duration_exceeds_promise.astype(int).eq(a.Late_delivery_risk)
a['cancellation_aware_rule']=(a.duration_exceeds_promise & ~a.canceled).astype(int)
a['aware_rule_agrees']=a.cancellation_aware_rule.eq(a.Late_delivery_risk)
rows=[]
for status,sub in a.groupby('Delivery Status',sort=True):
 rows.append({'status':status,'orders':len(sub),'late_labels':int(sub.Late_delivery_risk.sum()),'duration_exceeds_promise':int(sub.duration_exceeds_promise.sum()),'simple_rule_disagreements':int((~sub.simple_rule_agrees).sum()),'cancellation_aware_disagreements':int((~sub.aware_rule_agrees).sum())})
pd.DataFrame(rows).to_csv(R/'outcome_label_audit.csv',index=False)
out={'population':'development orders only; reserved test excluded','development_orders':len(a),'within_order_conflicts':conflicts,'simple_duration_rule_agreements':int(a.simple_rule_agrees.sum()),'simple_duration_rule_disagreements':int((~a.simple_rule_agrees).sum()),'simple_duration_rule_agreement_rate':float(a.simple_rule_agrees.mean()),'canceled_orders':int(a.canceled.sum()),'noncanceled_orders':int((~a.canceled).sum()),'cancellation_aware_agreements':int(a.aware_rule_agrees.sum()),'cancellation_aware_disagreements':int((~a.aware_rule_agrees).sum()),'status_counts':rows}
(R/'outcome_label_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
# The cluster sample selects whole customers, with order-count weighted loss means.
p=pd.read_csv(R/'enhanced_validation_predictions.csv');assert np.array_equal(p.y.to_numpy(),o.loc[v,'Late_delivery_risk'])
names=['Logistic full','Random forest','Gradient boosting'];loss=(p[names].to_numpy()-p.y.to_numpy()[:,None])**2
customers,inv=np.unique(o.loc[v,'Customer Id'].to_numpy(),return_inverse=True)
closs=np.zeros((len(customers),3));np.add.at(closs,inv,loss);sizes=np.bincount(inv)
rng=np.random.default_rng(645);means=np.empty((5000,3))
for j in range(5000):
 ix=rng.integers(0,len(customers),len(customers));means[j]=closs[ix].sum(axis=0)/sizes[ix].sum()
contr={}
for (i,n),(j,m) in itertools.combinations(enumerate(names),2):
 contr[n+' vs '+m]={'difference':float((loss[:,i]-loss[:,j]).mean()),'ci9833':np.quantile(means[:,i]-means[:,j],[.05/6,1-.05/6]).tolist()}
b={'replicates':5000,'seed':645,'clusters':len(customers),'validation_orders':len(p),'method':'paired whole-customer percentile bootstrap of Brier-loss differences; 98.333% intervals','contrasts':contr}
(R/'paired_brier_bootstrap_5000.json').write_text(json.dumps(b,indent=2),encoding='utf-8')
e=json.loads((R/'enhanced_results.json').read_text())
if e['customer_bootstrap'].get('paired_replicates',500)==500:e['customer_bootstrap']['paired9833_500_draws']=e['customer_bootstrap']['paired9833']
e['customer_bootstrap']['paired9833']={k:x['ci9833'] for k,x in contr.items()};e['customer_bootstrap']['paired_replicates']=5000;e['customer_bootstrap']['paired_seed']=645
(R/'enhanced_results.json').write_text(json.dumps(e,indent=2),encoding='utf-8')
print(json.dumps({'label_audit':out,'paired_bootstrap':b},indent=2),flush=True)
