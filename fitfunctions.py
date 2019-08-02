def MCMC_fit(spectrum,model,params,grid,ncores,lines):
    import emcee
    import lmfit

    result = model.fit(spectrum[lines],params,wl=grid,method="emcee",nan_policy='omit',\
                             calc_covar=True, fit_kws={'nwalkers':500,'steps':25})
    return result