import torch

def get_mass_max(x, p):
    ap = p.argmax(dim=2)
    a = torch.eye(p.shape[2], device=p.device)[ap]
    mom = torch.matmul(a.transpose(1, 2), x)
    pt = torch.sqrt(mom[:, :, 1]**2 + mom[:, :, 2]**2)
    eta = torch.asinh(mom[:, :, 3] / pt)
    phi = torch.atan2(mom[:, :, 2], mom[:, :, 1])
    eps = 0.0001
    m = torch.sqrt((mom[:, :, 0] * (1 + eps))**2 - mom[:, :, 1]**2 - mom[:, :, 2]**2 - mom[:, :, 3]**2 + eps)
    mom = torch.stack([pt, eta, phi, m], dim=-1)
    ap[x[:, :, 0] == 0] = -1
    return mom, ap

def get_mass_set(X, P):
    is_lead_P = P[:, :, 0]
    is_sublead_P = P[:, :, 1]
    is_ISR_P = P[:, :, 2]
    emptyjets = X[:, :, 3] == 0
    is_ISR_P[emptyjets] = 99
    jet_4p = X
    n_event = jet_4p.shape[0]
    n_jet = 8
    n_ISR = n_jet - 6
    n_gluino = 3
    v, i = torch.sort(is_ISR_P, dim=1, descending=True)
    ISR_threshold = v[:, 1]
    ISR_mask = (is_ISR_P >= ISR_threshold[:, None]).int()
    renorm_lead_P = is_lead_P
    renorm_lead_P = renorm_lead_P * (1 - ISR_mask)
    v, i = torch.sort(renorm_lead_P, dim=1, descending=True)
    lead_threshold = v[:, 2]
    lead_mask = (renorm_lead_P >= lead_threshold[:, None]).int()
    sublead_mask = 1 - lead_mask - ISR_mask
    a = torch.stack([lead_mask, sublead_mask, ISR_mask], dim=2)
    mom = torch.matmul(a, X)
    pt = torch.sqrt(mom[:, :, 1]**2 + mom[:, :, 2]**2)
    eta = torch.asinh(mom[:, :, 3] / pt)
    phi = torch.atan2(mom[:, :, 2], mom[:, :, 1])
    m = torch.sqrt(mom[:, :, 0]**2 - mom[:, :, 1]**2 - mom[:, :, 2]**2 - mom[:, :, 3]**2)
    mom = torch.stack([pt, eta, phi, m], dim=-1)
    ap = a.argmax(dim=2)
    ap[X[:, :, 0] == 0] = -1
    return mom, ap

if __name__ == "__main__":
    x = torch.Tensor([
        [1.6803, -6.7709, -0.1952],
        [-0.7437, -0.4924, -1.3779],
        [-0.6305, -0.5535, -1.4105],
        [-0.8678, -0.4070, -1.3154],
        [-0.9150, -0.4179, -1.1902],
        [-0.8382, -0.5540, -1.0531],
        [-0.9211, -0.5754, -0.8581],
        [-0.9274, -0.7717, -0.4650]
    ])
    P = torch.Tensor([
        [[1., 0., 0.],
         [0., 1., 0.],
         [1., 0., 0.],
         [1., 0., 0.],
         [0., 1., 0.],
         [0., 1., 0.],
         [0., 0., 1.],
         [0., 0., 1.]]
    ])
    P = torch.nn.Softmax(dim=2)(P)
    X = torch.randn(P.shape[0], P.shape[1], 4)
    get_mass_max(X, P)