from pygen.dml.lang import gendml
from pygen.dists import bernoulli, normal
import torch


@gendml
def bar(mu):
    gentrace(normal, (mu, 1), "a")

@gendml
def baz(mu):
    gentrace(normal, (mu, 1), "b")

@gendml
def foo(mu):
    branch = gentrace(bernoulli, (0.4,), "branch")

    if branch:
        gentrace(normal, (mu, 1), "x")
        gentrace(bar, (mu,), None) #NB: would be "u" if we had hierarchical addresses
    else:
        gentrace(normal, (mu, 1), "y")
        gentrace(baz, (mu,), None) #NB: would be "v" if we had hierarchical addresses

def get_expected_score(mu, branch, x, a, y, b):
    score = torch.tensor(0.0)
    
    if branch:
        score += torch.log(torch.tensor(0.4))
        score += normal.logpdf((mu, 1), x)
        score += normal.logpdf((mu, 1), a)
    else:
        score += torch.log(1.0 - torch.tensor(0.4))
        score += normal.logpdf((mu, 1), y)
        score += normal.logpdf((mu, 1), b)

    return score

def test_simulate():
    mu = 0.123
    trace = foo.simulate((mu,))

    assert trace.get_gen_fn() == foo
    assert trace.get_args() == (mu,)
    assert trace.get_retval() is None
    choices = trace.get_choices()
    assert len(choices) == 3

    # FIXME: only one of these branches is taken, assuming tests are run w/ deterministic seeds. clarify/fix this.
    branch = choices["branch"]
    if branch:
        assert "x" in choices
        assert torch.isclose(trace.get_score(), get_expected_score(mu, branch, choices["x"], choices["a"], None, None))
    else:
        assert "y" in choices
        assert torch.isclose(trace.get_score(), get_expected_score(mu, branch, None, None, choices["y"], choices["b"]))

def test_generate():
    mu = 0.123
    constraints = {"branch": torch.tensor(1.0)}
    (trace, log_weight) = foo.generate((mu,), constraints)
    choices = trace.get_choices()
    assert choices["branch"] == torch.tensor(1.0)
    assert "x" in choices
    assert "a" in choices
    assert "y" not in choices
    assert "b" not in choices
    assert len(choices) == 3

    mu = 0.123
    constraints = {"branch": torch.tensor(0.0)}
    (trace, log_weight) = foo.generate((mu,), constraints)
    choices = trace.get_choices()
    assert choices["branch"] == torch.tensor(0.0)
    assert "x" not in choices
    assert "a" not in choices
    assert "y" in choices
    assert "b" in choices
    assert len(choices) == 3

def test_update():
    mu = 0.123
    constraints = {"branch": torch.tensor(0.0)}
    (trace, log_weight) = foo.generate((mu,), constraints)
    choices = trace.get_choices()

    (new_trace, new_log_weight, discard) = trace.update((mu,), {"branch": torch.tensor(1.0)})

    assert len(discard) == 3
    assert "y" in discard
    assert "b" in discard

def test_regenerate():
    mu = 0.123
    constraints = {"branch": torch.tensor(0.0)}
    (trace, log_weight) = foo.generate((mu,), constraints)

    for i in range(10):
        prev_choices = trace.get_choices()
        prev_mu = mu

        mu = torch.distributions.normal.Normal(0, 1).sample()
        (trace, weight) = foo.regenerate(trace, (mu,), ("branch"))
        choices = trace.get_choices()

        expected_weight = torch.tensor(0.0)
        
        if choices["branch"]:
            assert torch.isclose(trace.get_score(), get_expected_score(mu, branch, choices["x"], choices["a"], None, None))
            assert "x" in choices
            assert "a" in choices
            assert "y" not in choices
            assert "b" not in choices

            if prev_choices["branch"]:
                expected_weight = normal.logpdf((mu, 1), choices["x"]) + normal.logpdf((mu, 1), choices["a"])
                expected_weight -= (normal.logpdf((prev_mu, 1), choices["x"]) + normal.logpdf((prev_mu, 1), choices["a"]))
            else:
                expected_weight = 0
        else:
            assert torch.isclose(trace.get_score(), get_expected_score(mu, branch, None, None, choices["y"], choices["b"]))
            assert "x" not in choices
            assert "a" not in choices
            assert "y" in choices
            assert "b" in choices

            if not prev_choices["branch"]:
                expected_weight = normal.logpdf((mu, 1), choices["y"]) + normal.logpdf((mu, 1), choices["b"])
                expected_weight -= (normal.logpdf((prev_mu, 1), choices["y"]) + normal.logpdf((prev_mu, 1), choices["b"]))
            else:
                expected_weight = 0

        assert torch.isclose(weight, expected_weight)
        
