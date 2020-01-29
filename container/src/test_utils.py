import pytest
import torch
import numpy as np
from utils import logit2probability


def test_logit2probability():
    input_logit = torch.tensor([-4.595119850135, -2.197224577336, -1.38629436112, 0.0, 1.38629436112])
    prob = torch.tensor([0.01, 0.1, 0.2, 0.5, 0.8])

    pred_prob = logit2probability(input_logit)

    assert np.allclose(prob, pred_prob)
