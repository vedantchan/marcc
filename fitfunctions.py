def teffpred(spectrum):
    fitter = lmfit.Model(gen_sampler)
    params = lmfit.Parameters()
    params.add('t',min=5000,max=80000,value=13000)
    params.add('l',min=6.5,max=9.5,value=8.)
    params.add('scale',min=0.1,max=5,value=1)
    params.add('trans',min=-5,max=5,value=0)
    params['l'].set(value = 8.,vary=False)
    params['trans'].set(value = 0,vary=True)
    result = fitter.fit(spectra[i][mask],params,wl=lamgrid[mask],method='powell',nan_policy='omit')
    return result