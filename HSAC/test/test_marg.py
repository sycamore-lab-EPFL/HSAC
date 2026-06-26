import matplotlib.pyplot as plt
import numpy as np
import torch

N = 100000

loc = torch.tensor([[2,1]],dtype=torch.float32)
scale = torch.tensor([[0.1,0.1]])
dist = torch.distributions.Normal(loc,scale)
samples = dist.sample((N,))  # (N, 1, 2)
vecs = samples / torch.norm(samples, dim=-1, keepdim=True)  # (N, 1, 2)
angle = torch.atan2(vecs[...,1], vecs[...,0])  # (N, 1)
bins = np.linspace(-np.pi, np.pi, 101)
x,*_ = plt.hist(angle.numpy(), bins=bins, density=True,label='Monte Carlo Samples')

n_test = 100
angle_test = torch.vstack([torch.cos(torch.linspace(-np.pi, np.pi, n_test,requires_grad=True)),
                            torch.sin(torch.linspace(-np.pi, np.pi, n_test,requires_grad=True))]).T  # (1000, 2)
angle_var = 1/(angle_test/dist.scale).pow(2).sum(dim=-1)
angle_loc = (angle_test*dist.loc/dist.scale.pow(2)).sum(dim=-1)*angle_var
angle_dist = torch.distributions.Normal(loc=torch.zeros_like(angle_loc), scale=torch.ones_like(angle_var.sqrt()))
# Took me a day to figure this out, good luck understanding it in the future
# Hint: integrate the bivariate normal over the radius, remember to use rdr in polar coordinates, and remember that (mu^TAd)^2/d^TAd \neq mu^T A mu
angle_prob = ((angle_loc*(1-torch.erf(-np.sqrt(1/2)*angle_loc/angle_var.sqrt()))*np.sqrt(np.pi/2) +
              angle_var.sqrt()*torch.exp(-0.5*(angle_loc.pow(2)/angle_var)))*
              angle_var.sqrt()/(2*torch.pi*(dist.variance.prod(dim=-1)).sqrt())*
              torch.exp(-0.5*((loc.pow(2)/dist.variance).sum(dim=-1) - (angle_test*loc/scale.pow(2)).sum(dim=-1).pow(2)*angle_var))
                          )

plt.plot(torch.linspace(-np.pi, np.pi, n_test).detach().numpy(), angle_prob.detach().numpy(),label='PDF')
plt.legend()
plt.xlabel('Angle (radians)')
plt.ylabel('Density')
plt.show()
pass