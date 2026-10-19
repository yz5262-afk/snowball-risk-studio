import unittest
import numpy as np
from dataclasses import replace
from snowball_engine import SnowballParams, generate_normals, price_snowball, price_many_spots, target_coupon_for_par

class EngineTests(unittest.TestCase):
    def test_scalar_vector_and_seed(self):
        p=SnowballParams(n_sims=500)
        z=generate_normals(p,42)
        np.testing.assert_array_equal(z,generate_normals(p,42))
        vector=price_many_spots(np.array([70.,100.,120.]),0.,z,p)
        scalar=[price_snowball(s,0.,z,p) for s in [70.,100.,120.]]
        np.testing.assert_allclose(vector,scalar)

    def test_coupon_reprices_target(self):
        p=SnowballParams(n_sims=1000)
        z=generate_normals(p,7)
        coupon=target_coupon_for_par(p,z,target_price=100.)
        self.assertAlmostEqual(price_snowball(p.S0,0.,z,p,coupon_rate=coupon),100.,places=8)

    def test_terminal_payoff(self):
        p=SnowballParams(n_sims=100)
        z=generate_normals(p,1)
        self.assertAlmostEqual(price_snowball(50.,p.maturity_years,z,p),50.)
        self.assertAlmostEqual(price_snowball(80.,p.maturity_years,z,p),p.face_value*(1+p.coupon*p.maturity_years))

if __name__=='__main__': unittest.main()
