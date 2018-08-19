import torch


def wctransform(alpha, cf, sf):

    cf = cf.double()
    Cc, Wc, Hc = cf.size(0), cf.size(1), cf.size(2)
    cfv = cf.view(Cc, -1)  # c x (h x w)

    c_mean = torch.mean(cf, 1) # perform mean for each row
    c_mean = c_mean.unsqueeze(1).expand_as(cf) # add dim and replicate mean on rows
    cf = cf - c_mean # subtract mean element-wise

    cCov = torch.mm(cf, cf.t()).div((Wc * Hc) - 1) # construct covariance matrix

    c_u, c_e, c_v = torch.svd(cCov, some=False) # singular value decomposition

    k_c = Cc
    for i in range(Cc):
        if c_e[i] < 0.00001:
            k_c = i
            break


    sf = sf.double()
    _, Ws, Hs = sf.size(0), sf.size(1), sf.size(2)
    sfv = sf.view(Cc, -1)

    s_mean = torch.mean(sf, 1)
    s_mean = s_mean.unsqueeze(1).expand_as(sf)
    sf = sf - s_mean

    sCov = torch.mm(sf, sf.t()).div((Ws * Hs) - 1)
    s_u, s_e, s_v = torch.svd(sCov, some=False)

    k_s = Cc # same number of channels ad content features
    for i in range(Cc):
        if s_e[i] < 0.00001:
            k_s = i
            break


    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:,0:k_c],torch.diag(c_d))
    step2 = torch.mm(step1,(c_v[:,0:k_c].t()))
    whiten_cF = torch.mm(step2, cf)

    s_d = (s_e[0:k_s]).pow(0.5)
    targetFeature = torch.mm(torch.mm(torch.mm(s_v[:,0:k_s],torch.diag(s_d)),(s_v[:,0:k_s].t())),whiten_cF)
    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)


    targetFeature = targetFeature.view_as(cf)
    ccsF = alpha * targetFeature + (1.0 - alpha) * cf
    ccsF = ccsF.float().unsqueeze(0)

    # outf.data.resize_(ccsF.size()).copy_(ccsF)

    return ccsF
