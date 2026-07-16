"""Reference для FlowMatchEulerDiscreteScheduler FLUX: дамп sigmas/timesteps."""
import json
import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler

FLUX = "models/black-forest-labs/FLUX.1-dev"


def calculate_shift(seq, base_seq=256, max_seq=4096, base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq - base_seq)
    b = base_shift - m * base_seq
    return seq * m + b


def main():
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(f"{FLUX}/scheduler")
    for n_steps, seq in [(28, 4096), (4, 256), (50, 1024)]:
        sigmas0 = np.linspace(1.0, 1 / n_steps, n_steps)
        mu = calculate_shift(seq)
        sch.set_timesteps(sigmas=sigmas0, mu=mu)
        print(f"N={n_steps} seq={seq} mu={mu:.6f}")
        print("  timesteps[:5]", [round(float(x), 4) for x in sch.timesteps[:5].tolist()])
        print("  sigmas[:5]   ", [round(float(x), 6) for x in sch.sigmas[:5].tolist()])
        print("  sigmas[-3:]  ", [round(float(x), 6) for x in sch.sigmas[-3:].tolist()])


if __name__ == "__main__":
    main()
