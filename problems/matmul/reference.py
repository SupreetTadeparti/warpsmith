import torch


def make_inputs(dims, gen, dist):
    M, N, K = dims["M"], dims["N"], dims["K"]

    def rand(shape):
        if dist == "uniform":
            return torch.rand(shape, generator=gen, device="cuda") * 2 - 1
        if dist == "small_int":
            return torch.randint(-4, 5, shape, generator=gen, device="cuda").float()
        if dist == "wide_range":
            # Magnitudes spanning 1e-3..1e3 with random sign: punishes reduced precision.
            mag = 10.0 ** (torch.rand(shape, generator=gen, device="cuda") * 6 - 3)
            sign = torch.randint(0, 2, shape, generator=gen, device="cuda").float() * 2 - 1
            return mag * sign
        raise ValueError(f"unknown distribution {dist!r}")

    return {"A": rand((M, K)), "B": rand((K, N))}


def reference(inputs, dims):
    return {"C": inputs["A"].double() @ inputs["B"].double()}


def tolerance_scale(inputs, dims):
    return {"C": inputs["A"].double().abs() @ inputs["B"].double().abs()}


def baseline(inputs, outputs, dims):
    torch.matmul(inputs["A"], inputs["B"], out=outputs["C"])


def flops(dims):
    return 2 * dims["M"] * dims["N"] * dims["K"]


def bytes_moved(dims):
    M, N, K = dims["M"], dims["N"], dims["K"]
    return 4 * (M * K + K * N + M * N)
