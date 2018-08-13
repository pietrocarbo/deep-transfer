import torch

# TODO
def wctransform(alpha, cf, sf, outf=None):

    cf = cf.double()
    sf = sf.double()
    C,W,H = cf.size(0), cf.size(1), cf.size(2)
    _,W1,H1 = sf.size(0), sf.size(1), sf.size(2)
    cFView = cf.view(C, -1)
    sFView = sf.view(C, -1)



    cFSize = cf.size()
    c_mean = torch.mean(cf, 1) # c x (h x w)
    c_mean = c_mean.unsqueeze(1).expand_as(cf)
    cf = cf - c_mean

    contentConv = torch.mm(cf, cf.t()).div(cFSize[1] - 1) + torch.eye(cFSize[0]).double()
    c_u,c_e,c_v = torch.svd(contentConv,some=False)

    k_c = cFSize[0]
    for i in range(cFSize[0]):
        if c_e[i] < 0.00001:
            k_c = i
            break

    sFSize = sf.size()
    s_mean = torch.mean(sf, 1)
    sf = sf - s_mean.unsqueeze(1).expand_as(sf)
    styleConv = torch.mm(sf, sf.t()).div(sFSize[1] - 1)
    s_u,s_e,s_v = torch.svd(styleConv,some=False)

    k_s = sFSize[0]
    for i in range(sFSize[0]):
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
