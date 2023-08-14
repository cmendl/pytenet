import unittest
import numpy as np
import pytenet as ptn


class TestMPO(unittest.TestCase):

    def test_orthonormalization(self):

        # create random matrix product operator
        d = 4
        D = [1, 10, 13, 14, 7, 1]
        mpo0 = ptn.MPO(np.random.randint(-2, 3, size=d), [np.random.randint(-2, 3, size=Di) for Di in D], fill='random')

        self.assertEqual(mpo0.bond_dims, D, msg='virtual bond dimensions')

        # density matrix on full Hilbert space
        rho = mpo0.as_matrix()

        # performing left-orthonormalization...
        cL = mpo0.orthonormalize(mode='left')

        self.assertLessEqual(mpo0.bond_dims[1], d**2,
            msg='virtual bond dimension can only increase by factor of "d^2" per site')

        for i in range(mpo0.nsites):
            self.assertTrue(ptn.is_qsparse(mpo0.A[i], [mpo0.qd, -mpo0.qd, mpo0.qD[i], -mpo0.qD[i+1]]),
                            msg='sparsity pattern of MPO tensors must match quantum numbers')

        rhoL = mpo0.as_matrix()
        # density matrix should now be normalized
        self.assertAlmostEqual(np.linalg.norm(rhoL, 'fro'), 1., delta=1e-12, msg='density matrix normalization')

        # density matrices before and after left-normalization must match
        # (up to normalization factor)
        self.assertTrue(np.allclose(cL*rhoL, rho, rtol=1e-10),
                        msg='density matrices before and after left-normalization must match')

        # check left-orthonormalization
        for i in range(mpo0.nsites):
            s = mpo0.A[i].shape
            assert s[0] == d and s[1] == d
            Q = mpo0.A[i].reshape((s[0]*s[1]*s[2], s[3]))
            self.assertTrue(np.allclose(Q.conj().T @ Q, np.identity(s[3]), rtol=1e-12),
                            msg='left-orthonormalization')

        # performing right-orthonormalization...
        cR = mpo0.orthonormalize(mode='right')

        self.assertLessEqual(mpo0.bond_dims[-2], d**2,
            msg='virtual bond dimension can only increase by factor of "d^2" per site')

        for i in range(mpo0.nsites):
            self.assertTrue(ptn.is_qsparse(mpo0.A[i], [mpo0.qd, -mpo0.qd, mpo0.qD[i], -mpo0.qD[i+1]]),
                            msg='sparsity pattern of MPO tensors must match quantum numbers')

        self.assertAlmostEqual(abs(cR), 1., delta=1e-12,
            msg='normalization factor must have magnitude 1 due to previous left-orthonormalization')

        rhoR = mpo0.as_matrix()
        # density matrices must match
        self.assertAlmostEqual(np.linalg.norm(rhoL - cR*rhoR), 0., delta=1e-10,
                               msg='density matrices after left- and right-orthonormalization must match')

        # check right-orthonormalization
        for i in range(mpo0.nsites):
            s = mpo0.A[i].shape
            assert s[0] == d and s[1] == d
            Q = mpo0.A[i].transpose((0, 1, 3, 2)).reshape((s[0]*s[1]*s[3], s[2]))
            self.assertTrue(np.allclose(Q.conj().T @ Q, np.identity(s[2]), rtol=1e-12),
                            msg='right-orthonormalization')


    def test_identity(self):

        # dimensions
        d = 3
        L = 6

        # physical quantum numbers
        qd = np.random.randint(-2, 3, size=d)

        # construct MPO representation of identity
        idop = ptn.MPO.identity(qd, L)

        self.assertTrue(np.array_equal(idop.as_matrix(), np.identity(d**L)),
            msg='MPO representation of identity')


    def test_from_opchains(self):

        # dimensions
        d = 4
        L = 5

        # physical quantum numbers
        qd = np.random.randint(-2, 3, size=d)

        # fictitious operator chains
        opchains = []
        n = np.random.randint(20)
        for _ in range(n):
            istart = np.random.randint(L)
            length = np.random.randint(1, L - istart + 1)
            oplist = [crandn((d, d)) for _ in range(length)]
            qD = np.random.randint(-2, 3, size=length-1)
            # enforce sparsity structure dictated by quantum numbers
            qDpad = np.pad(qD, 1, mode='constant')
            for i in range(length):
                mask = ptn.qnumber_outer_sum([qd + qDpad[i], -(qd + qDpad[i+1])])
                oplist[i] = np.where(mask == 0, oplist[i], 0)
            opchains.append(ptn.OpChain(oplist, qD, istart))

        # construct MPO representation corresponding to operator chains
        mpo0 = ptn.MPO.from_opchains(qd, L, opchains)

        for i in range(mpo0.nsites):
            self.assertTrue(ptn.is_qsparse(mpo0.A[i], [mpo0.qd, -mpo0.qd, mpo0.qD[i], -mpo0.qD[i+1]]),
                            msg='sparsity pattern of MPO tensors must match quantum numbers')

        # construct full Hamiltonian from operator chains, as reference
        Href = sum(opc.as_matrix(d, L) for opc in opchains)

        # compare
        self.assertTrue(np.allclose(mpo0.as_matrix(), Href, rtol=1e-10),
            msg='full merging of MPO must be equal to matrix representation of operator chains')


    def test_add(self):

        # physical quantum numbers
        qd = np.random.randint(-2, 3, size=3)

        # create random matrix product operators
        qD0 = [np.random.randint(-2, 3, size=Di) for Di in [1, 11, 15, 23, 18, 9, 1]]
        qD1 = [np.random.randint(-2, 3, size=Di) for Di in [1, 7, 23, 11, 17, 13, 1]]
        # leading and trailing (dummy) virtual bond quantum numbers must agree
        qD1[ 0] = qD0[ 0].copy()
        qD1[-1] = qD0[-1].copy()
        op0 = ptn.MPO(qd, qD0, fill='random')
        op1 = ptn.MPO(qd, qD1, fill='random')

        # MPO addition
        op = op0 + op1

        # reference calculation
        op_ref = op0.as_matrix() + op1.as_matrix()

        # compare
        self.assertTrue(np.allclose(op.as_matrix(), op_ref, rtol=1e-12),
            msg='addition two MPOs must agree with matrix representation')


    def test_add_singlesite(self):

        # separate test for a single site since implementation is a special case

        # physical quantum numbers
        qd = np.random.randint(-2, 3, size=5)

        # create random matrix product operators acting on a single site
        # leading and trailing (dummy) virtual bond quantum numbers
        qD = [np.array([-1]), np.array([-2])]
        op0 = ptn.MPO(qd, qD, fill='random')
        op1 = ptn.MPO(qd, qD, fill='random')

        # MPO addition
        op = op0 + op1

        # reference calculation
        op_ref = op0.as_matrix() + op1.as_matrix()

        # compare
        self.assertTrue(np.allclose(op.as_matrix(), op_ref, rtol=1e-12),
            msg='addition two MPOs must agree with matrix representation')


    def test_sub(self):

        # physical quantum numbers
        qd = np.random.randint(-2, 3, size=3)

        # create random matrix product operators
        qD0 = [np.random.randint(-2, 3, size=Di) for Di in [1, 11, 15, 23, 18, 9, 1]]
        qD1 = [np.random.randint(-2, 3, size=Di) for Di in [1, 7, 23, 11, 17, 13, 1]]
        # leading and trailing (dummy) virtual bond quantum numbers must agree
        qD1[ 0] = qD0[ 0].copy()
        qD1[-1] = qD0[-1].copy()
        op0 = ptn.MPO(qd, qD0, fill='random')
        op1 = ptn.MPO(qd, qD1, fill='random')

        # MPO subtraction
        op = op0 - op1

        # reference calculation
        op_ref = op0.as_matrix() - op1.as_matrix()

        # compare
        self.assertTrue(np.allclose(op.as_matrix(), op_ref, rtol=1e-12),
            msg='subtraction two MPOs must agree with matrix representation')


    def test_sub_singlesite(self):

        # separate test for a single site since implementation is a special case

        # physical quantum numbers
        qd = np.random.randint(-2, 3, size=5)

        # create random matrix product operators acting on a single site
        # leading and trailing (dummy) virtual bond quantum numbers
        qD = [np.array([-1]), np.array([-2])]
        op0 = ptn.MPO(qd, qD, fill='random')
        op1 = ptn.MPO(qd, qD, fill='random')

        # MPO subtraction
        op = op0 - op1

        # reference calculation
        op_ref = op0.as_matrix() - op1.as_matrix()

        # compare
        self.assertTrue(np.allclose(op.as_matrix(), op_ref, rtol=1e-12),
            msg='subtraction two MPOs must agree with matrix representation')


    def test_multiply(self):

        # physical quantum numbers
        qd = np.random.randint(-2, 3, size=3)

        # create random matrix product operators
        op0 = ptn.MPO(qd, [np.random.randint(-2, 3, size=Di) for Di in [1, 10, 13, 24, 17, 9, 1]], fill='random')
        op1 = ptn.MPO(qd, [np.random.randint(-2, 3, size=Di) for Di in [1, 8, 17, 11, 23, 13, 1]], fill='random')

        # MPO multiplication (composition)
        op = op0 @ op1

        # reference calculation
        op_ref = op0.as_matrix() @ op1.as_matrix()

        # compare
        self.assertTrue(np.allclose(op.as_matrix(), op_ref, rtol=1e-12),
            msg='composition of two MPOs must agree with matrix representation')


def crandn(size):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    # 1/sqrt(2) is a normalization factor
    return (np.random.standard_normal(size)
       + 1j*np.random.standard_normal(size)) / np.sqrt(2)


if __name__ == '__main__':
    unittest.main()
