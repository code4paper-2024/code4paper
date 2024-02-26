import matlab.engine

eng = matlab.engine.start_matlab()

matlab_code = """
std_err=AIDSvalid.VarName6;
std_err(std_err==0)=eps;
std_err=-log(std_err);
t=isoutlier(std_err,"percentiles",[1,99]);
std_err(t)=[];
[f,xi]=ksdensity(std_err,'Function','cdf');
std_err=AIDStest.VarName6;
std_err(std_err==0)=eps;
std_err=-log(std_err);
vq = interp1(xi,f,std_err,'spline');
ind=AIDStest.VarName1==AIDStest.VarName2;
save('AIDS_AUC.mat','ind','vq');
"""

eng.eval(matlab_code, nargout=0)
eng.quit()
