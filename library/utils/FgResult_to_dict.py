from types import NoneType
import numpy as np

# Convert JAX/NumPy arrays to lists for JSON serialization
def __to_serializable(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: __to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [__to_serializable(x) for x in obj]
    return obj

def FgResult_to_dict(result):
    data = {
        "iterations": result.iterations,
        #"returned_params": __to_serializable(result.returned_params), # Removed because it takes too much space
        "final_fidelity": float(result.final_fidelity) if type(result.final_fidelity) != NoneType else -1,
        "final_purity":  float(result.final_purity) if type(result.final_purity) != NoneType else -1,
        "optimized_trainable_parameters": __to_serializable(result.optimized_trainable_parameters),
        "fidelity_each_timestep": __to_serializable(np.array(result.fidelity_each_timestep)),
        #"state_each_timestep_real": __to_serializable(np.real(np.array(result.state_each_timestep))), # Removed because it takes too much space
        #"state_each_timestep_imag": __to_serializable(np.imag(np.array(result.state_each_timestep))),
    }

    return data