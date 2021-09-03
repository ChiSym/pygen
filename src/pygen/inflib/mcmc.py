import torch

def mh_custom_proposal(trace, proposal, proposal_args):
    proposal_args_forward = (trace, *proposal_args)
    (fwd_choices, fwd_weight, _) = proposal.propose(proposal_args_forward)
    (new_trace, log_weight, discard) = trace.update(trace.get_args(), fwd_choices)
    proposal_args_backward = (new_trace, *proposal_args)
    (bwd_weight, _) = proposal.assess(proposal_args_backward, discard)
    alpha = log_weight - fwd_weight + bwd_weight
    if torch.log(torch.rand(())) < alpha:
        # accept
        return (new_trace, True)
    else:
        # reject
        return (trace, False)
