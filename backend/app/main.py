from __future__ import annotations
import csv, io, math, os
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache
from typing import Optional
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from .accounts import router as accounts_router, current_user, store_upload

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
app = FastAPI(title='Metals Intelligence API', version='1.0.0', description='Gold and silver analytics based on the supplied daily futures datasets.')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173','http://127.0.0.1:5173'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
app.include_router(accounts_router)


def number(x):
    try: return float(str(x).replace(',', '').replace('%','').strip())
    except (TypeError, ValueError): return np.nan

@lru_cache(maxsize=2)
def load_asset(asset: str) -> pd.DataFrame:
    asset = asset.lower()
    if asset not in ('gold','silver'): raise HTTPException(400, 'Asset must be gold or silver.')
    candidates = sorted(p for p in DATA.glob('*.csv') if p.name.lower().startswith(asset))
    parts=[]
    for p in candidates:
        try:
            raw=pd.read_csv(p)
            date_col=next((c for c in raw if c.lower()=='date'),None)
            close_col=next((c for c in raw if c.lower() in ('close/last','price','close')),None)
            if not date_col or not close_col: continue
            out=pd.DataFrame()
            out['date']=pd.to_datetime(raw[date_col],errors='coerce')
            out['close']=raw[close_col].map(number)
            for key in ('open','high','low','volume'):
                col=next((c for c in raw if c.lower()==key or (key=='volume' and c.lower()=='vol.')),None)
                out[key]=raw[col].map(number) if col else np.nan
            out['source']=p.name
            parts.append(out)
        except Exception: continue
    if not parts: return pd.DataFrame(columns=['date','close','open','high','low','volume','source'])
    df=pd.concat(parts,ignore_index=True).dropna(subset=['date','close'])
    # Where source periods overlap, prioritize the newer futures export; each date remains one observation.
    df['priority']=df['source'].str.contains('Futures',case=False).astype(int)
    df=df.sort_values(['date','priority']).drop_duplicates('date',keep='last').sort_values('date').drop(columns='priority').reset_index(drop=True)
    return df

def enriched(asset):
    d=load_asset(asset).copy()
    if d.empty: return d
    d['return_pct']=d.close.pct_change()*100
    d['ma20']=d.close.rolling(20,min_periods=1).mean()
    d['ma50']=d.close.rolling(50,min_periods=1).mean()
    d['volatility20']=d.return_pct.rolling(20,min_periods=2).std()*np.sqrt(252)
    return d

def rows(df):
    df=df.replace({np.nan:None})
    result=[]
    for row in df.to_dict('records'):
        result.append({k:(v.isoformat()[:10] if isinstance(v,(pd.Timestamp,datetime)) else (float(v) if isinstance(v,(np.floating,np.integer)) else v)) for k,v in row.items()})
    return result

@app.get('/api/health')
def health(): return {'status':'ok','database':'CSV-backed analytics; PostgreSQL schema available for deployment'}

@app.get('/api/overview')
def overview():
    assets={}
    for a in ('gold','silver'):
        d=enriched(a)
        if d.empty: continue
        latest=d.iloc[-1]; prev=d.iloc[-2] if len(d)>1 else latest
        assets[a]={'latest':float(latest.close),'date':latest.date.strftime('%Y-%m-%d'),'change':float(latest.close-prev.close),'change_pct':float((latest.close/prev.close-1)*100) if prev.close else 0,'volatility20':None if pd.isna(latest.volatility20) else float(latest.volatility20),'coverage_start':d.date.min().strftime('%Y-%m-%d'),'coverage_end':d.date.max().strftime('%Y-%m-%d'),'observations':len(d),'source':latest.source}
    ratio=None
    if all(a in assets for a in ('gold','silver')): ratio=assets['gold']['latest']/assets['silver']['latest']
    corr=None
    if all(a in assets for a in ('gold','silver')):
        g=load_asset('gold').set_index('date').close.pct_change(); s=load_asset('silver').set_index('date').close.pct_change(); pair=pd.concat([g.rename('gold'),s.rename('silver')],axis=1).dropna(); corr=float(pair.gold.corr(pair.silver)) if len(pair)>2 else None
    return {'assets':assets,'gold_silver_ratio':ratio,'gold_silver_return_correlation':corr,'currency':'USD','unit':'per troy ounce','as_of':max((v['date'] for v in assets.values()),default=None),'provenance':'Supplied daily futures CSV files; overlapping dates use the futures export.'}

@app.get('/api/history/{asset}')
def history(asset:str, start:Optional[str]=None,end:Optional[str]=None, interval:str='daily'):
    d=enriched(asset)
    if start: d=d[d.date>=pd.to_datetime(start)]
    if end: d=d[d.date<=pd.to_datetime(end)]
    if interval not in ('daily','weekly','monthly','quarterly','yearly'): raise HTTPException(400,'Unsupported interval.')
    if interval!='daily' and not d.empty:
        d=d.set_index('date').resample({'weekly':'W','monthly':'MS','quarterly':'QS','yearly':'YS'}[interval]).agg({'close':'mean','open':'first','high':'max','low':'min','volume':'sum','return_pct':'sum','ma20':'mean','ma50':'mean','volatility20':'mean','source':'last'}).dropna(subset=['close']).reset_index()
    return {'asset':asset.lower(),'currency':'USD','unit':'per troy ounce','interval':interval,'count':len(d),'data':rows(d)}

@app.get('/api/analytics/{asset}')
def analytics(asset:str):
    d=enriched(asset)
    if d.empty:return {'asset':asset,'summary':{},'data':[]}
    returns=d.return_pct.dropna()
    return {'asset':asset,'summary':{'mean_daily_return_pct':float(returns.mean()),'annualized_volatility_pct':float(returns.std()*np.sqrt(252)) if len(returns)>1 else None,'min_close':float(d.close.min()),'max_close':float(d.close.max()),'observations':len(d)},'data':rows(d.tail(500))}

@app.get('/api/olap')
def olap(asset:str='all', year:Optional[int]=None, grain:str='month', pivot:bool=False):
    frames=[]
    for a in ('gold','silver'):
        if asset!='all' and a!=asset.lower():continue
        d=load_asset(a).copy(); d['asset']=a
        if year:d=d[d.date.dt.year==year]
        frames.append(d)
    if not frames:return {'data':[],'dimensions':['asset','period'],'measures':['average','minimum','maximum','observations']}
    d=pd.concat(frames).set_index('date')
    freq={'day':'D','month':'MS','quarter':'QS','year':'YS'}.get(grain)
    if not freq:raise HTTPException(400,'grain must be day, month, quarter, or year')
    result=d.groupby('asset').resample(freq).close.agg(['mean','min','max','count']).reset_index().rename(columns={'date':'period','mean':'average','min':'minimum','max':'maximum','count':'observations'})
    if pivot:
        p=result.pivot(index='period',columns='asset',values='average').reset_index()
        return {'data':rows(p),'operation':'pivot','grain':grain}
    return {'data':rows(result),'operation':'roll-up' if grain!='day' else 'drill-down','grain':grain}

@app.post('/api/forecast')
def forecast(payload:dict):
    asset=str(payload.get('asset','gold')).lower(); horizon=int(payload.get('horizon',5)); model=payload.get('model','random_forest')
    if horizon<1 or horizon>30:raise HTTPException(400,'Horizon must be 1–30 trading days.')
    d=load_asset(asset)
    if len(d)<120:raise HTTPException(422,'At least 120 daily observations are required to train and evaluate a model.')
    if model not in ('random_forest','baseline'):raise HTTPException(400,'Supported models: random_forest, baseline. XGBoost is not available in this installation.')
    values=d.close.to_numpy(float); lag=20
    X=np.array([[values[i-j] for j in range(1,lag+1)] for i in range(lag,len(values))]); y=values[lag:]
    split=max(int(len(y)*.8),1)
    from sklearn.ensemble import RandomForestRegressor
    if model=='random_forest':
        estimator=RandomForestRegressor(n_estimators=180,max_depth=8,min_samples_leaf=3,random_state=42,n_jobs=-1).fit(X[:split],y[:split])
        pred=estimator.predict(X[split:])
    else:
        pred=X[split:,0]
    actual=y[split:]; mae=float(np.mean(np.abs(actual-pred))); rmse=float(np.sqrt(np.mean((actual-pred)**2))); mape=float(np.mean(np.abs((actual-pred)/np.maximum(np.abs(actual),1e-8)))*100)
    train_dates=d.date.iloc[lag+split-1].strftime('%Y-%m-%d')
    trained=RandomForestRegressor(n_estimators=180,max_depth=8,min_samples_leaf=3,random_state=42,n_jobs=-1).fit(X,y) if model=='random_forest' else None
    context=list(values[-lag:]); forecasts=[]; day=d.date.iloc[-1]
    for _ in range(horizon):
        feat=np.array([context[-lag:][::-1]])
        nxt=float(trained.predict(feat)[0]) if trained else context[-1]
        context.append(nxt); day=day+timedelta(days=1)
        while day.weekday()>=5:day+=timedelta(days=1)
        forecasts.append({'date':day.strftime('%Y-%m-%d'),'price':nxt})
    return {'asset':asset,'model':model,'horizon':horizon,'training_end':train_dates,'evaluation':'chronological 80/20 holdout','metrics':{'MAE':mae,'RMSE':rmse,'MAPE_pct':mape},'holdout': [{'date':d.date.iloc[lag+split+offset+i].strftime('%Y-%m-%d'),'actual':float(a),'predicted':float(p)} for i,(a,p) in enumerate(zip(actual[-120:],pred[-120:])) for offset in [len(actual)-min(len(actual),120)]],'forecast':forecasts,'disclaimer':'Forecasts are estimates based on historical prices and are not financial advice.'}



@app.get('/api/olap/cube')
def olap_cube(grain: str = 'year', year: Optional[int] = None):
    """Return observed price cube at time × asset × observed-price-band grain."""
    if grain not in ('year','quarter','month','day'):
        raise HTTPException(400,'grain must be year, quarter, month, or day')
    if grain == 'day' and year is None:
        raise HTTPException(400,'Select a year before drilling down to daily observations.')
    frames=[]
    available_years=set()
    for asset in ('gold','silver'):
        d=load_asset(asset).copy()
        d['asset']=asset
        available_years.update(d.date.dt.year.astype(str).unique().tolist())
        d['daily_return_pct']=d.close.pct_change()*100
        # Price bands are tertiles of this metal's observed distribution, not externally defined markets.
        q1,q2=d.close.quantile([1/3,2/3]).tolist()
        d['price_band']=np.select([d.close<=q1,d.close<=q2],['Lower','Middle'],default='Upper')
        if year is not None:
            d=d[d.date.dt.year==year]
        if grain=='year': d['period']=d.date.dt.year.astype(str)
        elif grain=='quarter': d['period']=d.date.dt.to_period('Q').astype(str)
        elif grain=='month': d['period']=d.date.dt.to_period('M').astype(str)
        else: d['period']=d.date.dt.strftime('%Y-%m-%d')
        frames.append(d)
    all_data=pd.concat(frames,ignore_index=True)
    records=[]
    for (period,asset,band),g in all_data.groupby(['period','asset','price_band'],sort=True):
        g=g.sort_values('date'); rets=g.daily_return_pct.dropna()
        records.append({'period':period,'asset':asset,'price_band':band,'average_close':float(g.close.mean()),'min_close':float(g.close.min()),'max_close':float(g.close.max()),'observations':int(len(g)),'volume':None if g.volume.isna().all() else float(g.volume.sum()),'return_pct':float((g.close.iloc[-1]/g.close.iloc[0]-1)*100) if len(g)>1 else (float(rets.iloc[-1]) if len(rets) else None),'volatility_pct':float(rets.std()*np.sqrt(252)) if len(rets)>1 else None})
    periods=sorted(all_data.period.unique().tolist())
    return {'grain':grain,'year':year,'years':sorted(available_years),'periods':periods,'assets':['gold','silver'],'price_bands':['Lower','Middle','Upper'],'band_definition':'Within each asset, lower/middle/upper thirds of all observed closing prices in the supplied history.','dimensions':['time','asset','observed price band'],'measures':['average_close','min_close','max_close','return_pct','volatility_pct','observations','volume'],'data':records}

@app.get('/api/warehouse')
def warehouse():
    counts={a:len(load_asset(a)) for a in ('gold','silver')}
    return {'tables':[{'name':'fact_daily_price','type':'fact','rows':sum(counts.values()),'grain':'one asset per observed date'},{'name':'dim_date','type':'dimension','rows':None},{'name':'dim_asset','type':'dimension','rows':2},{'name':'dim_source','type':'dimension','rows':None}],'counts':counts,'validation':{'invalid_dates':0,'missing_close':0,'duplicate_asset_dates':0},'etl':'Ready — source CSV files loaded on request','schemas':['star','snowflake']}

@app.post('/api/upload/inspect')
async def inspect_upload(file:UploadFile=File(...), user:dict=Depends(current_user)):
    name=file.filename or ''
    if not name.lower().endswith(('.csv','.xlsx','.xls')):raise HTTPException(415,'Upload a CSV or Excel workbook.')
    blob=await file.read()
    if len(blob)>15*1024*1024:raise HTTPException(413,'Files must be 15 MB or smaller.')
    try:
        if name.lower().endswith('.csv'): df=pd.read_csv(io.BytesIO(blob)); sheet=None
        else:
            book=pd.ExcelFile(io.BytesIO(blob)); sheet=book.sheet_names[0]; df=pd.read_excel(io.BytesIO(blob),sheet_name=sheet)
    except Exception as e:raise HTTPException(400,f'Could not read dataset: {e}')
    duplicates=int(df.duplicated().sum());saved_id=store_upload(user['id'],name,sheet,len(df),len(df.columns),duplicates,blob)
    return {'filename':name,'worksheet':sheet,'rows':len(df),'columns':[{'name':str(c),'type':str(t),'missing':int(df[c].isna().sum())} for c,t in df.dtypes.items()],'duplicates':duplicates,'preview':df.head(10).replace({np.nan:None}).to_dict('records'),'preserved':True,'saved_upload_id':saved_id,'note':'Original upload is saved privately in your account.'}

@app.post('/api/upload/etl')
async def execute_etl(file:UploadFile=File(...)):
    """Normalize a user file to the project's date/OHLCV convention and return it as CSV."""
    name=file.filename or 'dataset.csv'; blob=await file.read()
    try:
        if name.lower().endswith('.csv'): source=pd.read_csv(io.BytesIO(blob))
        elif name.lower().endswith(('.xlsx','.xls')): source=pd.read_excel(io.BytesIO(blob))
        else: raise HTTPException(415,'Upload a CSV or Excel workbook.')
    except HTTPException: raise
    except Exception as e: raise HTTPException(400,f'Could not read dataset: {e}')
    normalized={str(c).strip().lower():c for c in source.columns}
    date_col=next((normalized[k] for k in ('date','datetime','timestamp') if k in normalized),None)
    close_col=next((normalized[k] for k in ('close/last','close','price') if k in normalized),None)
    if date_col is None or close_col is None: raise HTTPException(422,'ETL requires a date column and a close/last, close, or price column.')
    result=pd.DataFrame({'date':pd.to_datetime(source[date_col],errors='coerce')})
    for target,aliases in {'close':('close/last','close','price'),'open':('open',),'high':('high',),'low':('low',),'volume':('volume','vol.')}.items():
        col=next((normalized[k] for k in aliases if k in normalized),None)
        if col is not None: result[target]=source[col].map(number)
    before=len(result); result=result.dropna(subset=['date','close']).sort_values('date').drop_duplicates('date',keep='last')
    removed=before-len(result); result['date']=result.date.dt.strftime('%Y-%m-%d')
    output=io.StringIO(); result.to_csv(output,index=False)
    from fastapi.responses import Response
    safe=Path(name).stem.replace(' ','_')+'_processed.csv'
    return Response(output.getvalue(),media_type='text/csv',headers={'Content-Disposition':f'attachment; filename="{safe}"','X-Processed-Rows':str(len(result)),'X-Removed-Rows':str(removed)})
