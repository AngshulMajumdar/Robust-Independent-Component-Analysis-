import os, time, json, platform, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.datasets import load_digits, load_wine, load_breast_cancer, load_diabetes
from sklearn.decomposition import FastICA
from scipy.optimize import linear_sum_assignment
from skimage import data, color, transform
from picard import picard
warnings.filterwarnings('ignore')

R=5; M=8

def std_rows(S):
    S=S-S.mean(1,keepdims=True); d=S.std(1,keepdims=True); d[d<1e-10]=1
    return S/d

def real_tabular_sources(X,n,seed):
    rng=np.random.default_rng(seed); X=np.asarray(X,float); good=np.flatnonzero(X.std(0)>1e-10)
    feat=rng.choice(good,R,replace=False); idx=rng.choice(X.shape[0],min(n,X.shape[0]),replace=False)
    B=X[idx][:,feat].T
    return std_rows(np.vstack([B[j,rng.permutation(B.shape[1])] for j in range(R)]))

def natural_image_sources(n,seed):
    imgs=[data.camera(),data.moon(),data.coins(),data.clock(),color.rgb2gray(data.astronaut())]
    vec=[]
    for im in imgs:
        im=transform.resize(np.asarray(im,float),(128,128),anti_aliasing=True).ravel()
        vec.append(im)
    rng=np.random.default_rng(seed); idx=rng.choice(len(vec[0]),min(n,len(vec[0])),replace=False)
    return std_rows(np.vstack([v[idx] for v in vec]))

def mixing(seed):
    rng=np.random.default_rng(seed); Q,_=np.linalg.qr(rng.normal(size=(M,M))); V,_=np.linalg.qr(rng.normal(size=(R,R)))
    s=np.linspace(1,.4,R); A=Q[:,:R]@np.diag(s)@V.T; A/=np.linalg.norm(A,axis=0,keepdims=True)
    return A

def contam(X,eps,seed,mag=10):
    rng=np.random.default_rng(seed+9000); Y=X.copy(); k=int(round(eps*X.shape[1]));
    if k==0:return Y
    idx=rng.choice(X.shape[1],k,replace=False); D=rng.normal(size=(M,k)); D/=np.linalg.norm(D,axis=0,keepdims=True)+1e-12
    scale=np.median(np.linalg.norm(X,axis=0)); amp=mag*scale*rng.uniform(.8,1.2,k); Y[:,idx]+=D*amp
    return Y

def align_corr(S,T):
    S=std_rows(S); T=std_rows(T); C=np.abs(np.corrcoef(S,T)[:R,R:]); a,b=linear_sum_assignment(-C); return C[a,b].mean()

def amari(P):
    P=np.abs(P); r=(P/(P.max(1,keepdims=True)+1e-12)).sum(1)-1; c=(P/(P.max(0,keepdims=True)+1e-12)).sum(0)-1
    return (r.sum()+c.sum())/(2*R*(R-1))

def whiten(X,r=R):
    X0=X-X.mean(1,keepdims=True); C=X0@X0.T/X0.shape[1]; d,E=np.linalg.eigh(C); ix=np.argsort(d)[::-1][:r]; d=d[ix]; E=E[:,ix]
    K=np.diag(1/np.sqrt(np.maximum(d,1e-10)))@E.T; return K@X0,K

def fast(X,fun,seed,algorithm='parallel'):
    f=FastICA(n_components=R,fun=fun,algorithm=algorithm,whiten='unit-variance',max_iter=300,tol=1e-5,random_state=seed)
    S=f.fit_transform(X.T).T; return S,f.components_

def pic(X,ortho,seed):
    K,W,Y=picard(X,n_components=R,ortho=ortho,extended=True,whiten=True,max_iter=200,tol=1e-5,random_state=seed)
    return Y,W@K

def joint_diag(mats,max_sweeps=100,tol=1e-7):
    mats=np.array(mats,float); p=mats.shape[1]; V=np.eye(p)
    for _ in range(max_sweeps):
        mx=0
        for a in range(p-1):
            for b in range(a+1,p):
                g1=mats[:,a,a]-mats[:,b,b]; g2=mats[:,a,b]+mats[:,b,a]
                ton=np.sum(g1*g1-g2*g2); toff=2*np.sum(g1*g2); theta=.25*np.arctan2(toff,ton+1e-30)
                c,s=np.cos(theta),np.sin(theta); mx=max(mx,abs(s))
                if abs(s)<tol: continue
                G=np.array([[c,-s],[s,c]])
                mats[:,[a,b],:]=np.einsum('ij,kjl->kil',G.T,mats[:,[a,b],:])
                mats[:,:,[a,b]]=np.einsum('kij,jl->kil',mats[:,:,[a,b]],G)
                V[:,[a,b]]=V[:,[a,b]]@G
        if mx<tol:break
    return V

def jade(X,seed):
    Z,K=whiten(X); n=Z.shape[1]; mats=[]
    for i in range(R):
        for j in range(i,R):
            q=Z[i]*Z[j]; M=(Z*q)@Z.T/n
            M-=np.eye(R)*(1 if i==j else 0)
            M-=np.outer(np.eye(R)[i],np.eye(R)[j])+np.outer(np.eye(R)[j],np.eye(R)[i])
            mats.append((M+M.T)/2)
    V=joint_diag(mats); W=V.T@K; return W@(X-X.mean(1,keepdims=True)),W

def sobi(X,seed):
    Z,K=whiten(X); mats=[]
    for lag in [1,2,3,5,7,11,17]:
        C=Z[:,lag:]@Z[:,:-lag].T/(Z.shape[1]-lag); mats.append((C+C.T)/2)
    V=joint_diag(mats); W=V.T@K; return W@(X-X.mean(1,keepdims=True)),W

def fobi(X,seed):
    Z,K=whiten(X); q=np.sum(Z*Z,0); B=(Z*q)@Z.T/Z.shape[1]; _,V=np.linalg.eigh(B); W=V.T@K; return W@(X-X.mean(1,keepdims=True)),W

def proposed(X,seed):
    X0=X-X.mean(1,keepdims=True); norms=np.linalg.norm(X0,axis=0); keep=norms<=np.quantile(norms,.85)
    # robust rank-R subspace init from central samples
    U,_,_=np.linalg.svd(X0[:,keep],full_matrices=False); A=U[:,:R]; S=A.T@X0
    lam=.03; delta=max(1e-4,1e-3*np.median(np.linalg.norm(X0-A@S,axis=0)))
    prev=np.inf
    for it in range(40):
        Res=X0-A@S; w=1/(2*np.sqrt(np.sum(Res*Res,0)+delta*delta))
        # weighted source gradient
        L=2*np.linalg.norm(A,2)**2*np.max(w)+lam
        eta=.7/(L+1e-12)
        for _ in range(8):
            G=2*A.T@((A@S-X0)*w[None,:])+lam*np.tanh(S); G-=G.mean(1,keepdims=True); S-=eta*G; S-=S.mean(1,keepdims=True)
        SD=S*w[None,:]; A=(X0*w[None,:])@S.T@np.linalg.inv(SD@S.T+1e-6*np.eye(R))
        ns=np.linalg.norm(A,axis=0)+1e-12; A/=ns; S*=ns[:,None]
        obj=np.sum(np.sqrt(np.sum((X0-A@S)**2,0)+delta**2)-delta)+lam*np.log(np.cosh(np.clip(S,-20,20))).sum()
        if abs(prev-obj)/max(1,abs(prev))<1e-7:break
        prev=obj
    # ICA rotation within robust subspace using Picard on high-weight samples
    Res=X0-A@S; w=1/(2*np.sqrt(np.sum(Res*Res,0)+delta*delta)); good=w>=np.quantile(w,.25)
    try:
        K,B,Y=picard(S[:,good],n_components=R,ortho=False,extended=True,whiten=True,max_iter=200,tol=1e-5,random_state=seed)
        T=B@K; W=T@np.linalg.pinv(A); Sout=W@X0
    except Exception:
        W=np.linalg.pinv(A); Sout=W@X0
    return Sout,W

methods={
'Proposed IRLS-l12':proposed,
'FastICA-logcosh':lambda X,s:fast(X,'logcosh',s),
'FastICA-exp':lambda X,s:fast(X,'exp',s),
'FastICA-cube':lambda X,s:fast(X,'cube',s),
'Picard-orthogonal':lambda X,s:pic(X,True,s),
'Picard-nonorthogonal':lambda X,s:pic(X,False,s),
'FastICA-deflation-logcosh':lambda X,s:fast(X,'logcosh',s,'deflation'),
'FastICA-deflation-cube':lambda X,s:fast(X,'cube',s,'deflation'),
'JADE':jade,'SOBI':sobi,'FOBI':fobi}

dsets={'Digits':load_digits().data,'Wine':load_wine().data,'BreastCancer':load_breast_cancer().data,'Diabetes':load_diabetes().data,'NaturalImages':None}
epss=[0,.05,.1,.2,.3]; seeds=range(10); t0=time.time()

def one_config(dn,D,seed,eps):
    S=natural_image_sources(150,seed) if D is None else real_tabular_sources(D,150,seed)
    A=mixing(seed); X=A@S; Y=contam(X,eps,seed); out=[]
    for mn,fn in methods.items():
        st=time.time(); ok=True; err=''
        try:
            Sh,W=fn(Y,seed); corr=align_corr(S,Sh); ae=amari(W@A)
            if not np.isfinite(corr+ae): raise ValueError('nonfinite')
        except Exception as e:
            ok=False; corr=np.nan; ae=np.nan; err=f'{type(e).__name__}:{str(e)[:80]}'
        out.append(dict(dataset=dn,seed=seed,epsilon=eps,method=mn,mean_abs_corr=corr,amari_error=ae,runtime_sec=time.time()-st,success=ok,error=err,n_samples=S.shape[1],sensors=M,sources=R))
    return out

jobs=[(dn,D,seed,eps) for dn,D in dsets.items() for seed in seeds for eps in epss]
chunks=Parallel(n_jobs=8,backend='threading',verbose=10)(delayed(one_config)(*j) for j in jobs)
rows=[r for ch in chunks for r in ch]
res=pd.DataFrame(rows); res.to_csv('/mnt/data/ica_real_benchmark_full.csv',index=False)
sumry=res.groupby(['dataset','epsilon','method'],as_index=False).agg(mean_corr=('mean_abs_corr','mean'),std_corr=('mean_abs_corr','std'),mean_amari=('amari_error','mean'),std_amari=('amari_error','std'),runtime=('runtime_sec','mean'),success_rate=('success','mean'))
sumry.to_csv('/mnt/data/ica_real_benchmark_summary.csv',index=False)
over=res.groupby('method',as_index=False).agg(mean_corr=('mean_abs_corr','mean'),mean_amari=('amari_error','mean'),runtime=('runtime_sec','mean'),success_rate=('success','mean'))
over['rank_corr']=over.mean_corr.rank(ascending=False); over['rank_amari']=over.mean_amari.rank(); over['avg_rank']=(over.rank_corr+over.rank_amari)/2; over=over.sort_values('avg_rank'); over.to_csv('/mnt/data/ica_real_benchmark_overall.csv',index=False)
meta=dict(platform=platform.platform(),python=platform.python_version(),cpu_count=os.cpu_count(),elapsed_seconds=time.time()-t0,total_runs=len(res),protocol='5 real-source datasets; 8 sensors; 5 sources; dense samplewise contamination; magnitude 10x median clean sample norm; 10 seeds')
json.dump(meta,open('/mnt/data/ica_real_benchmark_machine.json','w'),indent=2)
print(over.to_string(index=False)); print(meta)
