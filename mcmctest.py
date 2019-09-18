import lmfit
import numpy as np

x = np.linspace(0,10,100);
y = 3*x + 2

def linear(x,m,c):
	return m*x + c

model = lmfit.Model(linear);
params = model.make_params(m = 1,c=1);

res = model.fit(y,params,x=x)

print(res);

f = open('output.txt','w');
f.write(lmfit.fit_report(res))
f.close();
