from DistanceQuery import BVH
import numpy as np
import scipy.special

def safe_div(num, denom):
    return num/np.where(denom==0.0, 1.0, denom)

def uniform_sphere(*shape, d):
    # this fucntion sample uniformly on the sphere
    res=np.random.randn(*shape, d)
    return res/np.linalg.norm(res, axis=-1, keepdims=True)

def uniform_ball(*shape, d):
    res=uniform_sphere(*shape, d=d)
    return (np.random.rand(*shape)**(1/d))[..., None]*res

def area_ball(d, dist):
    #dist is an array of shape (n, )
    return (np.pi ** (d/2.0) / scipy.special.gamma(d/2.0 + 1.0)) * (dist**d) #shape (n, )
 
def Laplace_sample(q: np.array, g, bvh: BVH, n_iter=1000, tol=1e-8):
    # q is a query of points of shape (n, 3)
    q=q.copy()
    n, d=q.shape
    indices=np.arange(n)
    for _ in range(n_iter):
        _, dist, _=bvh.closest_point(q[indices])
        mask=dist>tol
        if not mask.any():
            break
        indices=indices[mask]
        q[indices]=uniform_sphere(len(indices),d=d)*dist[mask][:, None]+q[indices]
    closest,_, _=bvh.closest_point(q)
    return g(closest) #shape (n, )

def control_grad_Laplace(q: np.array, g, bvh: BVH, num_samples=10000):
    _, dist, _=bvh.closest_point(q)
    n, d=q.shape

    outward_normal=uniform_sphere(n, num_samples, d=d)

    sample=q[:, None, :]+dist[:, None, None]*outward_normal
    estimation_sol=Laplace_sample(sample.reshape(-1, d), g, bvh).reshape(n, num_samples) #shape(n,num_samples)
    running_estimate=estimation_sol[:, :-1].cumsum(axis=1)/np.arange(1, num_samples)
    estimation_sol[:, 1:]-=running_estimate

    return safe_div(d, dist)[:, None]*(estimation_sol[..., None]*outward_normal).mean(axis=1) #shape(n, d)

def Laplace_sol(q: np.array, g, bvh: BVH, num_samples=10000, grad_samples=1000):
    n, d=q.shape
    outward_normal=uniform_sphere(n, num_samples, d=d) #shape(n, num_samples, d)
    _, dist, _ = bvh.closest_point(q)
    grad=control_grad_Laplace(q, g, bvh, num_samples=grad_samples) #shape (n, d)

    sample=q[:, None, :]+dist[:, None, None]*outward_normal #(n, num_samples, d)
    sample=Laplace_sample(sample.reshape(-1, d), g, bvh).reshape(n, num_samples) 

    prod=np.einsum('nd, nmd-> nm', grad, outward_normal)*dist[:, None]
    return (sample-prod).mean(axis=1)


def G_ball(x: np.array, y: np.array, R: np.array): #calculate the green harmonic function
    #x, y have to have of the same shape (n, d), r is a scalar
    #the dimension is supposed to be greater or equal than 2
    d=x.shape[1]
    r=np.linalg.norm(x-y, axis=-1)
    if d==2:
        return 1/(2*np.pi)*np.log(safe_div(R, r))
    return 1/(4*np.pi)*(safe_div(R-r, R*r))

def grad_green(x:np.array, y:np.array, R:np.array):
    d=x.shape[1]
    difference=y-x
    norm=np.linalg.norm(difference, axis=-1)
    if d==2:
        return difference/(2*np.pi)*((safe_div(1.0, norm**2)-safe_div(1.0, R**2))[:, None])
    return difference/(4*np.pi)*((safe_div(1.0, norm**3)-safe_div(1.0, R**3))[:, None])

def Poisson_sample(q: np.array, f, g, bvh: BVH, n_iter=1000, tol=1e-8):
    q=q.copy()
    n, d=q.shape
    res=np.zeros(n)
    indices=np.arange(n)
    for _ in range(n_iter):
        _, dist, _=bvh.closest_point(q[indices])
        mask=dist>tol
        if not mask.any():
            break
        indices, dist=indices[mask], dist[mask]
        y=uniform_ball(len(indices), d=d)*dist[:, None]+q[indices]
        res[indices]+=area_ball(d, dist)*f(y)*G_ball(q[indices], y, dist)
        q[indices]=uniform_sphere(len(indices),d=d)*dist[:, None]+q[indices]
    closest,_, _=bvh.closest_point(q)
    return res+g(closest)

def sample_grad_Poisson(q: np.array, f, g, bvh:BVH):
    _, dist, _=bvh.closest_point(q)
    n, d=q.shape
    normal=uniform_sphere(n, d=d)
    x=normal*dist[:, None]+q
    y=uniform_ball(n, d=d)*dist[:, None]+q
    return (safe_div(d, dist)*Poisson_sample(x, f, g, bvh))[:, None]*normal+(area_ball(d, dist)*f(y))[:, None]*grad_green(q, y, dist)