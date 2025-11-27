from cs336_basics.optimizer import SGD
import torch

def main():
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=100)
    for t in range(50):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights ** 2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.

if __name__ == "__main__":
    main()
    """
    Conclusion: lr=1, 10, 100 will make convergence faster and faster. lr=1000 will cause divergence.
    """

