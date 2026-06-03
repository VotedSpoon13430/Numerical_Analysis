import numpy as np
import time
import math

class Assignment4:
    def __init__(self):
        self.eps=1e-16#eps tol

    def solve_normal_equations(self,A,b):
        n=len(b)
        aug=np.column_stack([A,b])#aug matrix
        for col in range(n):
            max_idx=col+np.argmax(np.abs(aug[col:,col]))#pivot select
            if abs(aug[max_idx,col])<self.eps:
                return np.zeros(n,dtype=np.float64)#singular
            if max_idx!=col:
                aug[[col,max_idx]]=aug[[max_idx,col]]#row swap
            aug[col+1:,col:]-=aug[col+1:,col:col+1]/aug[col,col]*aug[col,col:]#eliminate
        x=np.zeros(n,dtype=np.float64)
        for i in range(n-1,-1,-1):
            x[i]=(aug[i,n]-aug[i,i+1:n]@x[i+1:])/aug[i,i]#back solve
        return x

    def least_squares_fit(self,A,b):
        ATA=A.T@A#normal eq
        ATb=A.T@b
        n=ATA.shape[0]
        reg=1e-14*np.trace(ATA)/n#ridge
        ATA.flat[::n+1]+=reg#diag add
        return self.solve_normal_equations(ATA,ATb)

    def build_chebyshev_matrix(self,x,degree):
        n=len(x)
        T=np.zeros((n,degree+1),dtype=np.float64)#basis
        T[:,0]=1.0#T0
        if degree>=1:
            T[:,1]=x#T1
        for d in range(2,degree+1):
            T[:,d]=2*x*T[:,d-1]-T[:,d-2]#recurrence
        return T

    def evaluate_chebyshev_vec(self,coef,x):
        x=np.atleast_1d(x)#vectorize
        T=self.build_chebyshev_matrix(x,len(coef)-1)
        return T@coef#poly eval

    def detect_function_type(self,x,y):
        n=len(y)
        if n<5:
            return'poly',1#fallback
        y_min,y_max=np.min(y),np.max(y)
        idx=np.argsort(x)#sort x
        x_s,y_s=x[idx],y[idx]

        if y_min>2.5 and y_max>100:#large growth
            try:
                m=y>2.72#log safe
                if np.sum(m)>=max(5,n*0.5):
                    xv,yv=x[m],y[m]
                    ly=np.log(yv)#log
                    m2=ly>1.0#loglog safe
                    if np.sum(m2)>=max(5,n*0.3):
                        xv2,ly2=xv[m2],ly[m2]
                        lly=np.log(ly2)#loglog
                        if not np.any(~np.isfinite(lly)):
                            sx,sy=np.sum(xv2),np.sum(lly)
                            sxx,sxy=np.sum(xv2*xv2),np.sum(xv2*lly)
                            den=len(xv2)*sxx-sx*sx
                            if abs(den)>1e-15:
                                a=(len(xv2)*sxy-sx*sy)/den#slope
                                b=(sy-a*sx)/len(xv2)#intercept
                                r=lly-(a*xv2+b)#residual
                                r2=1-np.sum(r*r)/(np.sum((lly-np.mean(lly))**2)+1e-12)#r2
                                if r2>0.85 and abs(a-1.0)<0.5:
                                    return'double_exp',1#f8
            except:
                pass

        if n>=10:
            dy=np.diff(y_s)#derivative
            if np.sum(np.abs(np.diff(np.sign(dy)))>0)>n*0.08:
                return'oscillatory',min(50,n//2)#many flips

        mi=np.argmax(y_s)#peak
        if 0<mi<n-1:
            r=y_max/(np.mean([y_s[0],y_s[-1]])+1e-10)#edge ratio
            if r>5:
                if np.all(np.diff(y_s[:mi+1])>=-1e-6)or np.all(np.diff(y_s[mi:])<=1e-6):
                    return'gaussian',min(10,n//4)#bell shape

        if y_min>1e-9 and y_max/y_min>8:
            try:
                ly=np.log(y)#log fit
                A=np.column_stack([x,np.ones(n)])
                c=self.least_squares_fit(A,ly)
                p=A@c#predict
                r2=1-np.sum((ly-p)**2)/(np.sum((ly-np.mean(ly))**2)+1e-12)
                if r2>0.88:
                    return'exp',1#exp
            except:
                pass

        xn=2*(x-x.min())/(x.max()-x.min()+1e-12)-1#normalize
        best,bic=1,float('inf')
        for d in range(min(15,n//3)):
            T=self.build_chebyshev_matrix(xn,d)
            c=self.least_squares_fit(T,y)
            r=np.sum((y-T@c)**2)/n#mse
            b=n*np.log(r+1e-12)+(d+1)*np.log(n)#bic
            if b<bic:
                bic,best=b,d
        return'poly',best#default

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        st=time.time()#start
        if a==b:
            return lambda x:f(a)#degenerate
        v1=f(a+0.25*(b-a));v2=f(a+0.75*(b-a))#probe
        oh=(time.time()-st)/2#overhead
        tb=maxtime*0.96#budget
        tl=tb-(time.time()-st)
        if oh<1e-6:n=min(100000,int(tl/max(oh,1e-8)*0.85))
        elif oh<1e-5:n=min(80000,int(tl/oh*0.85))
        elif oh<1e-3:n=min(30000,int(tl/oh*0.75))
        else:n=min(8000,int(tl/oh*0.65))
        n=max(500 if d<=3 else 1000,min(n,int(tl/max(oh,1e-8)*0.7)))#min samples
        h=n//2
        xs=np.unique(np.concatenate([np.linspace(a,b,h),a+(b-a)*0.5*(1-np.cos(np.pi*(2*np.arange(1,n-h+1)-1)/(2*(n-h))))]))#cheb grid
        xd,yd=[],[]
        for x in xs:
            if time.time()-st>tb:break#time stop
            try:
                y=f(x)
                if np.isfinite(y):
                    xd.append(x);yd.append(y)#collect
            except:
                pass
        if len(xd)<4:
            return lambda x:(v1+v2)/2#fallback
        xa,ya=np.array(xd),np.array(yd)
        t,deg=self.detect_function_type(xa,ya)#classify
        if t=='double_exp':
            return self.fit_double_exp_weighted(f,a,b,st,tb)#f8 path
        if t=='gaussian':
            return self.fit_log_space(xa,ya,a,b,deg)#gauss
        if t=='exp':
            return self.fit_log_space(xa,ya,a,b,min(d,10))#exp
        if t=='oscillatory':
            return self.fit_polynomial(xa,ya,min(deg,d,len(xa)//3),a,b)#osc
        return self.fit_polynomial(xa,ya,min(deg,d,len(xa)//4),a,b)#poly

    def fit_polynomial(self,x,y,deg,a,b):
        xn=2*(x-a)/(b-a)-1#normalize
        deg=max(1,min(deg,len(x)//3,60))#limit
        T=self.build_chebyshev_matrix(xn,deg)
        c=self.least_squares_fit(T,y)#coeffs
        def f(xn):
            xn=np.atleast_1d(xn)
            t=2*(xn-a)/(b-a)-1
            r=self.evaluate_chebyshev_vec(c,t)
            return r[0] if len(r)==1 else r
        return f

    def fit_log_space(self,x,y,a,b,deg):
        m=y>1e-10#positive
        xp,yp=x[m],y[m]
        if len(xp)<3:
            return lambda x:np.mean(y)#fallback
        ly=np.log(yp)#log
        xn=2*(xp-a)/(b-a)-1
        deg=min(deg,len(xp)//4,12)#cap
        T=self.build_chebyshev_matrix(xn,deg)
        c=self.least_squares_fit(T,ly)
        def f(x):
            x=np.atleast_1d(x)
            t=2*(x-a)/(b-a)-1
            v=np.clip(self.evaluate_chebyshev_vec(c,t),-700,700)#clip
            r=np.exp(v)
            return r[0] if len(r)==1 else r
        return f

    def fit_double_exp_weighted(self,f,a,b,st,tb):
        xs,ys=[],[]
        while time.time()-st<tb*0.75 and len(xs)<2500:
            try:
                u=np.random.random()
                x=a+(b-a)*(1-(1-u)**3)#right bias
                y=f(x)
                if np.isfinite(y)and y>1.0:
                    xs.append(x);ys.append(y)
            except:
                pass
        if len(xs)<30:
            c=f(a)
            return lambda x:c#safe const
        xs,ys=np.array(xs),np.array(ys)
        z=np.log(np.log(ys))#loglog
        def build(s,i):
            def g(x):
                x=min(max(x,a),b)#clamp
                v=s*x+i
                if v>709:return 1.7e308#overflow
                e=math.exp(v)
                if e>709:return 1.7e308
                return math.exp(e)
            return g
        w=ys/(ys.max()+1e-12)#weights
        mx,mz=np.sum(w*xs)/np.sum(w),np.sum(w*z)/np.sum(w)
        num,den=np.sum(w*(xs-mx)*(z-mz)),np.sum(w*(xs-mx)**2)
        if abs(den)<1e-18:
            return lambda x:f(a)#fallback
        return build(num/den,mz-(num/den)*mx)#final
##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1, 1, 1))
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1, 1, 1)))

        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1, 1, 1)
        nf = NOISY(1)(f)
        ass4 = Assignment4()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse = 0
        for x in np.linspace(0, 1, 1000):
            self.assertNotEquals(f(x), nf(x))
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1000
        print(mse)


if __name__ == "__main__":
    unittest.main()