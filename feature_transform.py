import torch


def decomposition(f, regularize=True):
    f = f.double()
    f_channels, f_width, f_height = f.size(0), f.size(1), f.size(2)
    fv = f.view(f_channels, -1)  # c x (h x w)

    f_mean = torch.mean(fv, 1) # perform mean for each row
    f_mean = f_mean.unsqueeze(1).expand_as(fv) # add dim and replicate mean on rows
    fv -= f_mean # subtract mean element-wise

    if regularize:
        f_covariance = torch.mm(fv, fv.t()).div((f_width * f_height) - 1) + torch.eye(f_channels).double() # construct regularized covariance matrix
    else:
        f_covariance = torch.mm(fv, fv.t()).div((f_width * f_height) - 1)
    f_u, f_e, f_v = torch.svd(f_covariance, some=False) # singular value decomposition

    f_c = f_channels
    for i in range(f_channels):
        if f_e[i] < 0.00001:
            f_c = i
            break

    return f_e, f_v, f_c, fv, f_mean


def zca_whitening(features):
    f_e, f_v, f_c, fv, _ = decomposition(features)
    f_d = (f_e[0:f_c]).pow(-0.5)

    w_step1 = torch.mm(f_v[:, 0:f_c], torch.diag(f_d))
    w_step2 = torch.mm(w_step1, (f_v[:, 0:f_c].t()))
    whitened = torch.mm(w_step2, fv)
    return whitened


def coloring(whitened_content_f, style_features):
    sf_e, sf_v, sf_c, _, sf_mean = decomposition(style_features, regularize=False)
    sf_d = (sf_e[0:sf_c]).pow(0.5)

    c_step1 = torch.mm(sf_v[:, 0:sf_c], torch.diag(sf_d))
    c_step2 = torch.mm(c_step1, sf_v[:, 0:sf_c].t())
    colored = torch.mm(c_step2, whitened_content_f)

    colored += sf_mean.resize_as_(colored)
    return colored

def wct(alpha, cf, s0f, s1f=None, beta=None):
    whitened_content_f = zca_whitening(cf)
    colored_s0f = coloring(whitened_content_f, s0f)

    if beta:
        s1f = s1f.double()
        colored_s1f = coloring(whitened_content_f, s1f)
        trasformed_f = beta * colored_s0f + (1.0 - beta) * colored_s1f
    else:
        trasformed_f = colored_s0f

    trasformed_f = trasformed_f.view_as(cf)
    out_f = alpha * cf.double() + (1.0 - alpha) * trasformed_f
    return out_f.float().unsqueeze(0)


def wctransform(alpha, cf, sf):

    cf = cf.double()
    c_channels, c_width, c_height = cf.size(0), cf.size(1), cf.size(2)
    cfv = cf.view(c_channels, -1)  # c x (h x w)

    c_mean = torch.mean(cfv, 1) # perform mean for each row
    c_mean = c_mean.unsqueeze(1).expand_as(cfv) # add dim and replicate mean on rows
    cfv = cfv - c_mean # subtract mean element-wise

    c_covm = torch.mm(cfv, cfv.t()).div((c_width * c_height) - 1) + torch.eye(c_channels).double() # construct covariance matrix
    c_u, c_e, c_v = torch.svd(c_covm, some=False) # singular value decomposition

    k_c = c_channels
    for i in range(c_channels):
        if c_e[i] < 0.00001:
            k_c = i
            break


    sf = sf.double()
    _, s_width, s_heigth = sf.size(0), sf.size(1), sf.size(2)
    sfv = sf.view(c_channels, -1)

    s_mean = torch.mean(sfv, 1)
    s_mean = s_mean.unsqueeze(1).expand_as(sfv)
    sfv = sfv - s_mean

    s_covm = torch.mm(sfv, sfv.t()).div((s_width * s_heigth) - 1)
    s_u, s_e, s_v = torch.svd(s_covm, some=False)

    s_k = c_channels # same number of channels ad content features
    for i in range(c_channels):
        if s_e[i] < 0.00001:
            s_k = i
            break


    c_d = (c_e[0:k_c]).pow(-0.5)
    s_d = (s_e[0:s_k]).pow(0.5)

    w_step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    w_step2 = torch.mm(w_step1, (c_v[:, 0:k_c].t()))
    whitened = torch.mm(w_step2, cfv)

    c_step1 = torch.mm(s_v[:, 0:s_k], torch.diag(s_d))
    c_step2 = torch.mm(c_step1, s_v[:, 0:s_k].t())
    colored = torch.mm(c_step2, whitened)

    target_features = colored + s_mean.resize_as_(colored)
    target_features = target_features.view_as(cf)

    ccsf = alpha * target_features + (1.0 - alpha) * cf
    ccsf = ccsf.float().unsqueeze(0)

    return ccsf


def wc_transform(alpha, cf, sf, s1f=None, beta=None):

    # content image whitening
    cf = cf.double()
    c_channels, c_width, c_height = cf.size(0), cf.size(1), cf.size(2)
    cfv = cf.view(c_channels, -1)  # c x (h x w)

    c_mean = torch.mean(cfv, 1) # perform mean for each row
    c_mean = c_mean.unsqueeze(1).expand_as(cfv) # add dim and replicate mean on rows
    cfv = cfv - c_mean # subtract mean element-wise

    c_covm = torch.mm(cfv, cfv.t()).div((c_width * c_height) - 1) + torch.eye(c_channels).double() # construct covariance matrix
    c_u, c_e, c_v = torch.svd(c_covm, some=False) # singular value decomposition

    k_c = c_channels
    for i in range(c_channels):
        if c_e[i] < 0.00001:
            k_c = i
            break
    c_d = (c_e[0:k_c]).pow(-0.5)

    w_step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    w_step2 = torch.mm(w_step1, (c_v[:, 0:k_c].t()))
    whitened = torch.mm(w_step2, cfv)

    # style image coloring
    sf = sf.double()
    _, s_width, s_heigth = sf.size(0), sf.size(1), sf.size(2)
    sfv = sf.view(c_channels, -1)

    s_mean = torch.mean(sfv, 1)
    s_mean = s_mean.unsqueeze(1).expand_as(sfv)
    sfv = sfv - s_mean

    s_covm = torch.mm(sfv, sfv.t()).div((s_width * s_heigth) - 1)
    s_u, s_e, s_v = torch.svd(s_covm, some=False)

    s_k = c_channels # same number of channels ad content features
    for i in range(c_channels):
        if s_e[i] < 0.00001:
            s_k = i
            break
    s_d = (s_e[0:s_k]).pow(0.5)

    c_step1 = torch.mm(s_v[:, 0:s_k], torch.diag(s_d))
    c_step2 = torch.mm(c_step1, s_v[:, 0:s_k].t())
    colored = torch.mm(c_step2, whitened)

    cs0_features = colored + s_mean.resize_as_(colored)
    cs0_features = cs0_features.view_as(cf)

    # additional style coloring
    if beta:
        sf = s1f
        sf = sf.double()
        _, s_width, s_heigth = sf.size(0), sf.size(1), sf.size(2)
        sfv = sf.view(c_channels, -1)

        s_mean = torch.mean(sfv, 1)
        s_mean = s_mean.unsqueeze(1).expand_as(sfv)
        sfv = sfv - s_mean

        s_covm = torch.mm(sfv, sfv.t()).div((s_width * s_heigth) - 1)
        s_u, s_e, s_v = torch.svd(s_covm, some=False)

        s_k = c_channels
        for i in range(c_channels):
            if s_e[i] < 0.00001:
                s_k = i
                break
        s_d = (s_e[0:s_k]).pow(0.5)

        c_step1 = torch.mm(s_v[:, 0:s_k], torch.diag(s_d))
        c_step2 = torch.mm(c_step1, s_v[:, 0:s_k].t())
        colored = torch.mm(c_step2, whitened)

        cs1_features = colored + s_mean.resize_as_(colored)
        cs1_features = cs1_features.view_as(cf)

        target_features = beta * cs0_features + (1.0 - beta) * cs1_features
    else:
        target_features = cs0_features

    ccsf = alpha * target_features + (1.0 - alpha) * cf
    ccsf = ccsf.float().unsqueeze(0)
    return ccsf
