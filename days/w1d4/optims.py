import torch

class SGD:
    def __init__(self, params, lr=0.01, momentum=0, dampening=0, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.b = [None] * len(self.params)
    def zero_grad(self):
        for param in self.params:
            param.grad = None
    def step(self):
        with torch.no_grad():
            for i, param in enumerate(self.params):
                gt = param.grad
                gt += self.weight_decay * param
                if self.momentum:
                    if self.b[i] is not None:
                        self.b[i] = self.momentum * self.b[i] + (1 - self.dampening) * gt
                    else:
                        self.b[i] = gt
                    gt = self.b[i]
                param -= self.lr * gt

class RMSProp:
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0):
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum

        self.v = [0] * len(self.params)
        self.b = [0] * len(self.params)
    def zero_grad(self):
        for param in self.params:
            param.grad = None
    def step(self):
        with torch.no_grad():
            for i, param in enumerate(self.params):
                gt = param.grad
                gt += self.weight_decay * param
                self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * gt**2  
                self.b[i] = self.momentum * self.b[i] + gt / (torch.sqrt(self.v[i]) + self.eps)
                param -= self.lr * self.b[i]

class Adam:
    def __init__(self, params, lr=0.001, betas=(.9, .999), eps=1e-08, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [0] * len(self.params)
        self.v = [0] * len(self.params)
        self.t = 1
    def zero_grad(self):
        for param in self.params:
            param.grad = None
    def step(self):
        with torch.no_grad():
            for i, param in enumerate(self.params):
                gt = param.grad
                gt += self.weight_decay * param
                self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * gt  
                self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * gt**2  
                m_norm = self.m[i] / (1-self.betas[0]**self.t)
                v_norm = self.v[i] / (1-self.betas[1]**self.t)
                param -= self.lr * m_norm/(torch.sqrt(v_norm)+self.eps)
            self.t += 1