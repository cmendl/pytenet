import time
import autoray as ar
from autoray import numpy as np
import torch
import pytenet as ptn


def test_opchain_to_matrix():

    torch.set_default_dtype(torch.float64)

    for backend in ["numpy", "torch"]:
        with ar.backend_like(backend):

            # physical quantum numbers
            qsite = [0, -1, 1]

            opids = [ 6,  3,  5,  4]
            qnums = [ 0, -1,  0,  2,  0]
            coeff = 0.7
            chain = ptn.OpChain(opids, qnums, coeff, 1)
            assert chain.length == len(opids)

            # random local operators
            rng = ar.do("random.default_rng", int(time.time()))
            opmap = { opid: ptn.crandn(2 * (len(qsite),), rng) for opid in range(3, 7) }
            # enforce sparsity pattern of local operators in the chain according to quantum numbers.
            for i, opid in enumerate(opids):
                mask = np.array(ptn.qnumber_outer_sum([
                            qsite, ptn.neg_qnumbers(qsite), [qnums[i]], [-qnums[i+1]]]))[:, :, 0, 0]
                opmap[opid] = np.where(mask == 0, opmap[opid], 0)

            # reference matrix representation
            mat_ref = coeff * np.identity(1)
            for opid in opids:
                mat_ref = np.kron(mat_ref, opmap[opid])

            # compare
            assert np.allclose(chain.to_matrix(opmap), mat_ref)
