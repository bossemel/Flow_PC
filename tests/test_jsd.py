import scipy.stats
from eval.metrics import js_divergence
import unittest
import random
import numpy as np
import torch 


class Test_JSD(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_JSD, self).__init__(*args, **kwargs)
        self.obs = 10000
        self.mv_gaussian_1 = scipy.stats.multivariate_normal(mean=[0, 0], cov=[[1., 0],
                                                                               [0, 1.]])
        self.samples_1 = self.mv_gaussian_1.rvs(self.obs)

    def test_one_pdf_same_samples(self):
        X_in_p = torch.from_numpy(self.mv_gaussian_1.pdf(self.samples_1))
        jsd_X_in_p = js_divergence(X_in_p, X_in_p, X_in_p, X_in_p)
        self.assertEqual(jsd_X_in_p, 0)

    def test_one_pdf_different_samples(self):
        samples_2 = self.mv_gaussian_1.rvs(self.obs)

        X_in_p = torch.from_numpy(self.mv_gaussian_1.pdf(self.samples_1))
        X_in_q = torch.from_numpy(self.mv_gaussian_1.pdf(self.samples_1))
        Y_in_p = torch.from_numpy(self.mv_gaussian_1.pdf(samples_2))
        Y_in_q = torch.from_numpy(self.mv_gaussian_1.pdf(samples_2))
        jsd_X_in_p_q = js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q)
        self.assertTrue(jsd_X_in_p_q >= 0 and jsd_X_in_p_q <= 0.01)

    def test_different_cov(self):
        mv_gaussian_2 = scipy.stats.multivariate_normal(mean=[0, 0], cov=[[1., 0.5],
                                                                          [0.5, 1.]])
        samples_2 = mv_gaussian_2.rvs(self.obs)

        X_in_p = torch.from_numpy(self.mv_gaussian_1.pdf(self.samples_1))
        X_in_q = torch.from_numpy(mv_gaussian_2.pdf(self.samples_1))
        Y_in_p = torch.from_numpy(self.mv_gaussian_1.pdf(samples_2))
        Y_in_q = torch.from_numpy(mv_gaussian_2.pdf(samples_2))
        jsd_X_Y = js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q)
        self.assertTrue(jsd_X_Y >= 0.01 and jsd_X_Y <= 0.5)

    def test_different_mean(self):
        mv_gaussian_2 = scipy.stats.multivariate_normal(mean=[10000, 10000], cov=[[10000., 0.5],
                                                                                  [0.5, 10000.]])
        samples_2 = mv_gaussian_2.rvs(self.obs)
        X_in_p = torch.from_numpy(self.mv_gaussian_1.pdf(self.samples_1))
        X_in_q = torch.from_numpy(mv_gaussian_2.pdf(self.samples_1))
        Y_in_p = torch.from_numpy(self.mv_gaussian_1.pdf(samples_2))
        Y_in_q = torch.from_numpy(mv_gaussian_2.pdf(samples_2))
        jsd_X_Y = js_divergence(X_in_p, X_in_q, Y_in_p, Y_in_q)
        self.assertTrue(jsd_X_Y >= 0.5 and jsd_X_Y <= 1)