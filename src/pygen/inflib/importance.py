from pygen.choice_trie import MutableChoiceTrie
import torch

def importance_sampling_custom_proposal(
        model, model_args, observations,
        proposal, proposal_args,
        num_samples, verbose=False):

    traces = []
    log_weights = []
    for i in range(num_samples):
        if verbose:
            print(f"sample: {i} of {num_samples}")
        (proposal_choices, proposal_weight, _) = proposal.propose(proposal_args)
        constraints = MutableChoiceTrie.copy(observations)
        constraints.update(proposal_choices)
        (trace, model_weight) = model.generate(model_args, constraints)
        traces.append(trace)
        log_weights.append(model_weight - proposal_weight)
    log_total_weight = torch.logsumexp(torch.tensor(log_weights), 0)
    log_ml_estimate = log_total_weight - torch.log(torch.tensor(num_samples))
    log_normalized_weights = torch.tensor(log_weights) - log_total_weight
    return (traces, log_normalized_weights, log_ml_estimate)

def importance_resampling_custom_proposal(
        model, model_args, observations,
        proposal, proposal_args,
        num_samples, verbose=False):
    (proposal_choices, proposal_weight, _) = proposal.propose(proposal_args)
    constraints = MutableChoiceTrie.copy(observations)
    constraints.update(proposal_choices)
    (model_trace, model_weight) = model.generate(model_args, constraints)
    log_total_weight = model_weight - proposal_weight
    for i in range(num_samples-1):
        if verbose:
            print(f"sample: {i} of {num_samples}")
        (proposal_choices, proposal_weight, _) = proposal.propose(proposal_args)
        constraints = MutableChoiceTrie.copy(observations)
        constraints.update(proposal_choices)
        (cand_model_trace, model_weight) = model.generate(model_args, constraints)
        log_weight = model_weight - proposal_weight
        log_total_weight = torch.logsumexp(torch.tensor([log_total_weight, log_weight]), 0)
        if torch.rand(()) < torch.exp(log_weight - log_total_weight):
            model_trace = cand_model_trace
    log_ml_estimate = log_total_weight - torch.log(torch.tensor(num_samples))
    return (model_trace, log_ml_estimate)
