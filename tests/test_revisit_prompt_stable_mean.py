"""Fixed reduction order must preserve the mathematical group means."""
import torch

from models.revisit_prompt_numerics import stable_group_mean


def test_means_keep_group_ids_empty_groups_and_gradient():
    index = torch.tensor([2,0,2,0])
    x = torch.tensor([[2.,4.],[1.,5.],[4.,8.],[3.,9.]], requires_grad=True)
    actual = stable_group_mean(x, index)
    torch.testing.assert_close(actual, torch.tensor([[2.,7.],[0.,0.],[3.,6.]]), rtol=0, atol=0)
    actual.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x,.5), rtol=0, atol=0)


def test_empty_input_keeps_feature_dimensions():
    result = stable_group_mean(torch.zeros(0,128), torch.zeros(0,dtype=torch.long))
    assert result.shape == (0,128)
