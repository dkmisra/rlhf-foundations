import torch 

from llm.moe import MixtureOfExpert


def main():
    dim = 32
    moe = MixtureOfExpert(num_experts=10, dim=dim, top_k=4)

    x = torch.randn(3, 10, dim)
    out, loss = moe(x, return_loss=True)

    print(f"Out is {out.shape} and load balancing loss is {loss:.2f}")


if __name__ == "__main__":
    main()    
