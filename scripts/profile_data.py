"""Regenerate the non-identifying raw DataCo quality audit."""
import argparse, hashlib, json
from pathlib import Path
import pandas as pd

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--csv', required=True)
    a.add_argument('--out', required=True)
    args=a.parse_args()
    path=Path(args.csv); out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(path,encoding='latin1')
    missing=d.isna().sum()
    summary={'rows':len(d),'columns':len(d.columns),'orders':int(d['Order Id'].nunique()),
             'missing':{k:int(v) for k,v in missing.items()},
             'target':{str(k):int(v) for k,v in d['Late_delivery_risk'].value_counts().items()},
             'shipping':{str(k):int(v) for k,v in d['Shipping Mode'].value_counts().items()},
             'order_shipping':{str(k):int(v) for k,v in d.drop_duplicates('Order Id')['Shipping Mode'].value_counts().items()},
             'within_order_target_mixed':int((d.groupby('Order Id')['Late_delivery_risk'].nunique()>1).sum()),
             'within_order_mode_mixed':int((d.groupby('Order Id')['Shipping Mode'].nunique()>1).sum()),
             'min_date':str(pd.to_datetime(d['order date (DateOrders)']).min()),
             'max_date':str(pd.to_datetime(d['order date (DateOrders)']).max()),
             'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (out/'source_profile.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    audit=missing.rename('missing_count').to_frame()
    audit['missing_pct']=100*audit['missing_count']/len(d)
    audit['fill_pct']=100-audit['missing_pct']
    audit.index.name='variable';audit.to_csv(out/'raw_missingness.csv')
    print('Rows:',len(d),'Orders:',summary['orders'],'SHA-256:',summary['sha256'])

if __name__=='__main__':main()
