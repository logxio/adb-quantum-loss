"""Exact small bond Born simulation of the S1 midpoint QCBM circuit."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import optimize, stats

SEED = 20261118


def circuit_mps(parameters, qubits, tolerance=1e-13):
    """Site zero is the least significant bit, as in the S1 state vector."""
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape != (2 * qubits,) or not np.all(np.isfinite(parameters)):
        raise ValueError("Expected two finite angles per qubit")
    cores = [np.array([[[1.], [0.]]]) for _ in range(qubits)]
    largest_discarded = 0.
    for layer in range(2):
        for j in range(qubits):
            c, s = np.cos(parameters[layer * qubits + j] / 2), np.sin(parameters[layer * qubits + j] / 2)
            cores[j] = np.einsum("st,atb->asb", np.array([[c, -s], [s, c]]), cores[j])
        for j in range(qubits - 1):
            joined = np.einsum("asb,btc->astc", cores[j], cores[j + 1])
            joined[:, 1, :, :] = joined[:, 1, ::-1, :]
            left, _, _, right = joined.shape
            a, singular, b = np.linalg.svd(joined.reshape(left * 2, 2 * right), full_matrices=False)
            rank = max(1, int(np.sum(singular > tolerance)))
            if rank < len(singular):
                largest_discarded = max(largest_discarded, float(np.max(singular[rank:])))
            cores[j] = a[:, :rank].reshape(left, 2, rank)
            cores[j + 1] = (singular[:rank, None] * b[:rank]).reshape(rank, 2, right)
    return cores, largest_discarded


def right_environments(cores):
    out = [None] * (len(cores) + 1)
    out[-1] = np.ones((1, 1))
    for j in range(len(cores) - 1, -1, -1):
        a = cores[j]
        out[j] = np.einsum("asc,bsd,cd->ab", a, a, out[j + 1], optimize=True)
    return out


def observed_probabilities(cores, bits):
    """Contract Born probabilities, summing out every bit encoded as minus one."""
    bits = np.asarray(bits, dtype=int)
    if bits.ndim != 2 or bits.shape[1] != len(cores) or np.any((bits < -1) | (bits > 1)):
        raise ValueError("Expected a matrix of binary or missing bits")
    left = np.ones((len(bits), 1, 1))
    for j, a in enumerate(cores):
        next_left = np.zeros((len(bits), a.shape[2], a.shape[2]))
        for bit in (0, 1):
            choose = (bits[:, j] == -1) | (bits[:, j] == bit)
            v = a[:, bit, :]
            next_left[choose] += np.einsum("ab,nac,cd->nbd", v, left[choose], v, optimize=True)
        left = next_left
    return left[:, 0, 0]


def encode_observations(u, bits_per_country=3):
    u = np.asarray(u, dtype=float)
    if u.ndim != 2 or np.any((u[np.isfinite(u)] < 0) | (u[np.isfinite(u)] > 1)):
        raise ValueError("Expected a PIT matrix, with NaN for missing countries")
    bins = 2 ** bits_per_country
    category = np.minimum((np.nan_to_num(u, nan=0.) * bins).astype(int), bins - 1)
    bits = ((category[:, :, None] >> np.arange(bits_per_country)) & 1).reshape(len(u), -1)
    bits[np.repeat(np.isnan(u), bits_per_country, axis=1)] = -1
    return bits


class NaiveQCBMMPS:
    """The original two layer RY and CNOT Born machine, without dense states."""
    name = "Naive Discretized QCBM Copula"

    def __init__(self, u, bits_per_country=3, maxiter=180, parameters=None):
        u = np.asarray(u, dtype=float)
        self.d, self.bits = u.shape[1], int(bits_per_country)
        self.qubits = self.d * self.bits
        self.bins = 2 ** self.bits
        self.states = 2 ** self.qubits
        encoded = encode_observations(u, self.bits)
        encoded = encoded[np.any(encoded != -1, axis=1)]
        if not len(encoded):
            raise ValueError("At least one observed PIT coordinate is required")
        encoded, counts = np.unique(encoded, axis=0, return_counts=True)
        weights = counts / counts.sum()

        def objective(angles):
            cores, _ = circuit_mps(angles, self.qubits)
            probability = observed_probabilities(cores, encoded)
            return float(-weights @ np.log(np.maximum(probability, 1e-12)))

        begin = time.perf_counter()
        if parameters is None:
            fitted = optimize.minimize(objective, np.full(2 * self.qubits, np.pi / 2), method="COBYLA", options={"maxiter": maxiter, "rhobeg": .35})
            parameters = fitted.x
            self.train_nll = float(fitted.fun)
            self.optimization_evaluations = int(fitted.nfev)
            self.optimizer_message = str(fitted.message)
        else:
            self.train_nll = objective(parameters)
            self.optimization_evaluations = 0
            self.optimizer_message = "Provided circuit parameters"
        self.parameters = np.asarray(parameters, dtype=float).tolist()
        self.cores, self.largest_discarded_singular_value = circuit_mps(self.parameters, self.qubits)
        self.right = right_environments(self.cores)
        self.norm_squared = float(self.right[0][0, 0])
        self.fit_seconds = time.perf_counter() - begin
        if abs(self.norm_squared - 1.) > 1e-10:
            raise ArithmeticError("Born circuit norm changed")

    def logpdf(self, u):
        return np.full(len(u), np.nan)

    def observed_probability(self, u):
        return observed_probabilities(self.cores, encode_observations(u, self.bits)) / self.norm_squared

    def sample(self, n, seed=SEED):
        if int(n) != n or n < 1:
            raise ValueError("Expected a positive sample count")
        uniforms = stats.qmc.Sobol(self.qubits, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(n))))[:n]
        left = np.ones((n, 1)) / np.sqrt(self.norm_squared)
        bits = np.empty((n, self.qubits), dtype=int)
        for j, a in enumerate(self.cores):
            candidates = [left @ a[:, bit, :] for bit in (0, 1)]
            probabilities = np.column_stack([np.einsum("na,ab,nb->n", x, self.right[j + 1], x, optimize=True) for x in candidates])
            probabilities = np.maximum(probabilities, 0.)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            bit = (uniforms[:, j] >= probabilities[:, 0]).astype(int)
            bits[:, j] = bit
            left = np.where(bit[:, None] == 0, candidates[0], candidates[1])
            left /= np.sqrt(np.maximum(probabilities[np.arange(n), bit], np.finfo(float).tiny))[:, None]
        categories = np.sum(bits.reshape(n, self.d, self.bits) * (2 ** np.arange(self.bits)), axis=2)
        return (categories + .5) / self.bins

    def exact_marginals(self):
        out = []
        for country in range(self.d):
            bits = np.full((self.bins, self.qubits), -1, dtype=int)
            bits[:, country * self.bits:(country + 1) * self.bits] = (np.arange(self.bins)[:, None] >> np.arange(self.bits)) & 1
            mass = observed_probabilities(self.cores, bits) / self.norm_squared
            mass = np.maximum(mass, 0.)
            out.append(mass / mass.sum())
        return np.asarray(out)

    def exact_marginal_ks(self):
        mass = self.exact_marginals()
        support = (np.arange(self.bins) + .5) / self.bins
        right_cdf = np.cumsum(mass, axis=1)
        return np.maximum(np.max(np.abs(right_cdf - support), axis=1), np.max(np.abs(right_cdf - mass - support), axis=1))

    def metadata(self):
        return {"qubits": self.qubits, "bits_per_country": self.bits, "layers": 2, "support_max": 1 - .5 / self.bins, "uniform_marginal_ks_exact": self.exact_marginal_ks().tolist(), "maximum_bond_dimension": max(a.shape[2] for a in self.cores), "norm_squared": self.norm_squared, "largest_discarded_singular_value": self.largest_discarded_singular_value, "fit_seconds": self.fit_seconds, "training_nll": self.train_nll, "optimization_evaluations": self.optimization_evaluations, "optimizer_message": self.optimizer_message, "training_likelihood": "Observed bit Born mass, summing missing countries exactly", "continuous_density": "Undefined for the atomic midpoint distribution", "sampling": "Exact conditional Born sampling with one scrambled Sobol coordinate per bit", "parameters": self.parameters, "core_storage_bytes": sum(a.nbytes for a in self.cores)}


def validate():
    import probe_s1_baselines as s1

    rng = np.random.default_rng(SEED)
    u = rng.uniform(size=(15, 2))
    reference = s1.NaiveQCBM(u)
    trials = []
    for i in range(3):
        angles = rng.uniform(-np.pi, np.pi, 12)
        model = NaiveQCBMMPS(u, parameters=angles)
        states = np.arange(64)
        bits = (states[:, None] >> np.arange(6)) & 1
        mps_born = observed_probabilities(model.cores, bits)
        dense_born = reference.born(angles)
        error = float(np.max(np.abs(mps_born - dense_born)))
        assert error < 1e-12
        expected = np.column_stack([(dense_born.reshape(8, 8).sum(axis=0)), (dense_born.reshape(8, 8).sum(axis=1))]).T
        marginal_error = float(np.max(abs(model.exact_marginals() - expected)))
        assert marginal_error < 1e-12
        samples = model.sample(65536)
        counts = np.bincount(np.sum((samples * 8).astype(int) * [1, 8], axis=1), minlength=64) / len(samples)
        sampling_error = float(np.max(np.abs(counts - dense_born)))
        assert sampling_error < .005
        trials.append({"case": i, "born_max_error": error, "marginal_max_error": marginal_error, "sample_mass_max_error": sampling_error, "maximum_bond": model.metadata()["maximum_bond_dimension"]})
    sparse = rng.uniform(size=(12, 15))
    sparse[rng.uniform(size=sparse.shape) < .25] = np.nan
    big = NaiveQCBMMPS(sparse, maxiter=180)
    sample = big.sample(4096)
    assert sample.shape == (4096, 15) and sample.max() <= .9375 and sample.min() >= .0625
    assert np.all(np.abs(big.exact_marginals().sum(axis=1) - 1) < 1e-12)
    result = {"dense_crosschecks": trials, "forty_five_qubit_observed_training": big.metadata(), "all_checks_passed": True}
    path = Path(__file__).resolve().parents[1] / "data/s3_qcbm_mps_checks.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(path), "dense_max_error": max(r["born_max_error"] for r in trials), "large_fit_seconds": big.fit_seconds, "max_bond": result["forty_five_qubit_observed_training"]["maximum_bond_dimension"]}))
    return result


if __name__ == "__main__":
    validate()
