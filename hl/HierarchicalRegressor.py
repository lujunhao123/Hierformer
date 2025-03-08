import numpy as np
import pandas as pd
import math
from scipy.linalg import block_diag
from scipy.stats import gmean
from statsmodels.tsa.stattools import acf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import copy
import torch
import torch.nn as nn

import torch
from hl.utils import matrix_tensor_multiplication
from typing import Tuple
import pickle
from hl.utils import quantileLoss
from os.path import exists

import hl.utils as utils
import hl.TreeClass as tc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class H_Regressor():
    """A Hierarchical regressor class object to encapsulate hierarchical predictive learning related functions."""

    def __init__(self, tree_obj: tc.Tree,
                       forecasting_method: str,
                       reconciliation_method: str,
                       **kwargs) -> None:

        self.tree = tree_obj
        self.output_length = self.tree.n
        self.scale_divider = None
        self.scale_shift = None
        #self.input_with_only_leaves = None
        self.hierarchical_forecasting_methods = ['base', 'multitask', 'OLS', 'BU', 'STR', 'hVAR', 'sVAR', 'COV',
                                                 'kCOV']
        self.hierarchical_reconciliation_methods = ['None', 'OLS', 'BU', 'STR', 'hVAR', 'sVAR', 'COV', 'kCOV']
        self.alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.75
        if forecasting_method not in self.hierarchical_forecasting_methods:
            raise ValueError('Invalid hierarchical forecasting method. Possible methods are : base, multitask, OLS, '
                             'BU, STR, hVAR, sVAR, COV, kCOV')
        else:
            self.hierarchical_forecasting_method = forecasting_method
        if reconciliation_method not in self.hierarchical_reconciliation_methods:
            raise ValueError('Invalid reconciliation method. Possible methods are : None, OLS, BU, STR, hVAR, sVAR, COV'
                             ', kCOV')
        else:
            self.reconciliation_method = reconciliation_method
        self.alpha_learn_Q = nn.Parameter(torch.tensor(0.5).clamp(0, 1))


    ###########################下面定义协调方法的实施########################  coherency_constraint是核心

    def coherency_constraint(self, y_diff=None, **kwargs):
        """Calculating the covariance matrix from a given reconciliation method."""
        reconciliation_method = kwargs['method'] if 'method' in kwargs else self.hierarchical_forecasting_method
        reconciliation_method_in = 'None' if reconciliation_method == 'multitask' else reconciliation_method
        output = kwargs['output'] if 'output' in kwargs else False
        # print(reconciliation_method)
        if reconciliation_method == 'BU':  # bottom up
            # Initializing the G matrix with zeros
            G = np.zeros((self.tree.m, self.tree.n))
            for i, elem in enumerate(self.tree.leaves_label):
                G[i, :] = [1 if id == elem else 0 for id in self.tree.y_ID]
            SG = np.matmul(self.tree.S, G)

            if output:
                return SG
            else:
                self.SG = SG
                self.SG_T = torch.tensor(self.SG)
                return
        else:
            sigma = self.covariance_matrix_calc(reconciliation_method_in, y_diff)
            sigma_inv = np.linalg.inv(sigma)
            SsigmaS = np.matmul(np.matmul(self.tree.S.T, sigma_inv), self.tree.S)
            SG = np.matmul(self.tree.S, np.matmul(np.matmul(np.linalg.inv(SsigmaS), self.tree.S.T), sigma_inv))
            if output:
                return SG
            else:
                self.sigma = sigma
                self.sigma_inv = sigma_inv
                self.SG = SG
                self.SG_T = torch.tensor(self.SG)

    def coherency_constraint_Q(self, y_diff_Q05=None, y_diff_Q95=None, **kwargs):
        """Calculating the covariance matrix from a given reconciliation method."""
        reconciliation_method = kwargs['method'] if 'method' in kwargs else self.hierarchical_forecasting_method
        reconciliation_method_in = 'None' if reconciliation_method == 'multitask' else reconciliation_method
        output = kwargs['output'] if 'output' in kwargs else False
        # print(reconciliation_method)
        if reconciliation_method == 'BU':  # bottom up
            # Initializing the G matrix with zeros
            G = np.zeros((self.tree.m, self.tree.n))
            for i, elem in enumerate(self.tree.leaves_label):
                G[i, :] = [1 if id == elem else 0 for id in self.tree.y_ID]
            SG = np.matmul(self.tree.S, G)

            if output:
                return SG
            else:
                self.SG = SG
                self.SG_Q05_T = torch.tensor(self.SG)
                self.SG_Q95_T = torch.tensor(self.SG)
                return
        else:
            sigma_Q05 = self.covariance_matrix_calc(reconciliation_method_in, y_diff_Q05)
            sigma_Q95 = self.covariance_matrix_calc(reconciliation_method_in, y_diff_Q95)
            sigma_inv_Q05 = np.linalg.inv(sigma_Q05)
            sigma_inv_Q95 = np.linalg.inv(sigma_Q95)

            SsigmaS_Q05 = np.matmul(np.matmul(self.tree.S.T, sigma_inv_Q05), self.tree.S)
            SsigmaS_Q95 = np.matmul(np.matmul(self.tree.S.T, sigma_inv_Q95), self.tree.S)

            SG_Q05 = np.matmul(self.tree.S, np.matmul(np.matmul(np.linalg.inv(SsigmaS_Q05), self.tree.S.T), sigma_inv_Q05))
            SG_Q95 = np.matmul(self.tree.S,np.matmul(np.matmul(np.linalg.inv(SsigmaS_Q95), self.tree.S.T), sigma_inv_Q95))


            if output:
                return SG_Q05,SG_Q95
            else:
                self.sigma_Q05 = sigma_Q05
                self.sigma_Q95 = sigma_inv_Q95
                self.sigma_inv_Q05 = sigma_inv_Q05
                self.sigma_inv_Q95 = sigma_inv_Q95
                self.SG_Q05 = SG_Q05
                self.SG_Q95 = SG_Q95
                self.SG_Q05_T = torch.tensor(self.SG_Q05)
                self.SG_Q95_T = torch.tensor(self.SG_Q95)

    def covariance_matrix_calc(self, reconciliation_method, y_diff_in=None):
        """Calculating the covariance matrix from a given reconciliation method."""

        R, Dhvar = None, None

        if y_diff_in is None:
            # Ordinary least squares default
            sigma = np.eye(self.output_length)
        else:
            if reconciliation_method == 'OLS':  # ordinary least squares
                sigma = np.eye(self.output_length)

            elif reconciliation_method == 'STR':  # structurally weighted least squares (WLS)
                leaves_maping = [1 if id in self.tree.leaves_label else 0 for i, id in enumerate(self.tree.y_ID)]
                leaves_var = np.var(y_diff_in[leaves_maping])
                sigma = leaves_var * np.diag(np.sum(self.tree.S, axis=1))

            elif reconciliation_method == 'hVAR':  # heterogeneous variance WLS
                if np.shape(y_diff_in)[1] == 1:
                    print("Warning: not enough samples to get a forecast error variance estimate per node")
                    var = np.var(y_diff_in)
                else:
                    var = np.var(y_diff_in, axis=1)
                sigma = np.diag(var)

            # homogeonous variance WLS uni-dimensional
            elif reconciliation_method == 'sVAR' and self.tree.dimension != 'spatiotemporal':
                svar = np.zeros(self.tree.n)
                for aggr_lvl in self.tree.k_level_map.values():
                    # Map the nodes in and not in that level of aggregation
                    map_nodes_aggrlvl = [i for i, id in enumerate(self.tree.y_ID) if id in aggr_lvl]
                    map_nodes_notinaggrlvl = [i for i, id in enumerate(self.tree.y_ID) if id not in aggr_lvl]
                    # Drop the rows not belonging to that level of aggregation
                    y_diff_aggrlvl = np.delete(y_diff_in, map_nodes_notinaggrlvl, axis=0)
                    # Estimate the common variance amongst this (homogenous) level of aggregation
                    node_var = np.var(y_diff_aggrlvl)
                    # Append the svar vector
                    svar[map_nodes_aggrlvl] = node_var
                sigma = np.diag(svar)

            # homogeonous variance WLS multi-dimensional
            elif reconciliation_method == 'sVAR' and self.tree.dimension == 'spatiotemporal':
                svar = np.zeros(self.tree.n)
                for S_aggr_lvl in self.tree.spatial.k_level_map.values():
                    for T_aggr_lvl in self.tree.temporal.k_level_map.values():
                        # Map the nodes in and not in that level of aggregation
                        map_nodes_aggrlvl = [i for i, id in enumerate(self.tree.y_ID) if
                                             id[0] in S_aggr_lvl and id[1] in T_aggr_lvl]
                        map_nodes_notinaggrlvl = [i for i, id in enumerate(self.tree.y_ID) if
                                                  i not in map_nodes_aggrlvl]
                        # Drop the rows not belonging to that level of aggregation
                        y_diff_aggrlvl = np.delete(y_diff_in, map_nodes_notinaggrlvl, axis=0)
                        # Estimate the common variance amongst this (homogenous) level of aggregation
                        node_var = np.var(y_diff_aggrlvl)
                        # Append the svar vector
                        svar[map_nodes_aggrlvl] = node_var
                sigma = np.diag(svar)

            elif reconciliation_method == 'COV':  # full covariance weighted least squares
                hvar = np.sqrt(np.var(y_diff_in, axis=1))
                Dhvar = np.diag(hvar)
                R = np.corrcoef(y_diff_in)
                R = np.nan_to_num(R, copy=True, nan=0.0, posinf=None, neginf=None)
                sigma = np.matmul(np.matmul(Dhvar, R), Dhvar)
                # sigma = np.cov(y_diff, bias=True)  # equivalent

            elif reconciliation_method == 'kCOV':  # k-level covariance weighted least squares
                hvar = np.sqrt(np.var(y_diff_in, axis=1))
                Dhvar = np.diag(hvar)
                R = self._to_klvl_block(np.corrcoef(y_diff_in))
                R = np.nan_to_num(R, copy=True, nan=0.0, posinf=None, neginf=None)
                sigma = np.matmul(np.matmul(Dhvar, R), Dhvar)
                # sigma = self._to_klvl_block(np.cov(y_diff, bias=True))  # equivalent

            elif reconciliation_method == 'multitask' or reconciliation_method == 'None':
                R, Dhvar = None, None
                sigma = np.eye(self.output_length)
            else:
                print('The reconciliation method has not been well defined. Review class initialization to avoid this '
                      'message.')
                R, Dhvar = None, None
                sigma = np.eye(self.output_length)

        # We verify if the covariance matrix is singular and proceed to shrink it if it is
        sigma = self.sigma_shrinkage_wrapper(sigma, R, Dhvar, reconciliation_method)
        return sigma

    def _to_klvl_block(self, M, **kwargs):
        """Enforces matrix M to be a block matrix according to k_level structure"""

        block_dim = kwargs['dimension'] if 'dimension' in kwargs else None
        # If the input tree is uni-dimensional, convert M to block format the following way
        if self.tree.dimension != 'spatiotemporal':
            # Looping over tree k_levels
            for aggr_lvl in self.tree.k_level_map.values():
                # Identifying nodes belonging to the same k-aggregation level
                node_map = [1 if id in aggr_lvl else 0 for i, id in enumerate(self.tree.y_ID)]
                # Fixing to 0 the elements not belonging to that block in the M matrix
                for i, elem in enumerate(node_map):
                    if elem == 1:
                        M[i, :] = M[i, :]*node_map
                        M[:, i] = M[:, i]*node_map
        elif block_dim is not None:
            # If the input tree is spatio-temporal & kwarg dimension is specified, convert M to block format along the
            # dimensional block specified
            if block_dim == 'temporal':
                k_lvl_vals_considered = self.tree.temporal.k_level_map.values()
                id_i = 1
            elif block_dim == 'spatial':
                k_lvl_vals_considered = self.tree.spatial.k_level_map.values()
                id_i = 0
            for T_aggr_lvl in k_lvl_vals_considered:
                # Map the nodes in that multi-level of aggregation
                bin_nodes_aggrlvl = [1 if id[id_i] in T_aggr_lvl else 0 for i, id in enumerate(self.tree.y_ID)]
                # Fixing to 0 the elements not belonging to that block in the M matrix
                for i, elem in enumerate(bin_nodes_aggrlvl):
                    if elem == 1:
                        M[i, :] = M[i, :] * bin_nodes_aggrlvl
                        M[:, i] = M[:, i] * bin_nodes_aggrlvl
            # If the input tree is spatio-temporal, convert M to block format along both spatio & temporal blocks only
        elif self.tree.dimension == 'spatiotemporal' and block_dim is None:
            for S_aggr_lvl in self.tree.spatial.k_level_map.values():
                for T_aggr_lvl in self.tree.temporal.k_level_map.values():
                    # Map the nodes in that multi-level of aggregation
                    bin_nodes_aggrlvl = [1 if id[0] in S_aggr_lvl and id[1] in T_aggr_lvl else 0 for i, id in
                                         enumerate(self.tree.y_ID)]
                    # Fixing to 0 the elements not belonging to that block in the M matrix
                    for i, elem in enumerate(bin_nodes_aggrlvl):
                        if elem == 1:
                            M[i, :] = M[i, :] * bin_nodes_aggrlvl
                            M[:, i] = M[:, i] * bin_nodes_aggrlvl
        return M

    def sigma_shrinkage_wrapper(self, sigma_in, R_in, Dhvar, reconciliation_method):
        """We shrink the correlation matrix for COV and kCOV reconciliation methods.
        The wrapper additionally verifies if the covariance matrix is singular and replaces it with the identity matrix
        if singular by design."""

        if np.linalg.det(sigma_in) == 0:
            if reconciliation_method not in ['COV', 'kCOV']:
                print("Warning the covariance matrix is Singular by design in method: " + str(reconciliation_method))
                sigma = np.eye(self.output_length)
                return sigma

        if reconciliation_method in ['COV', 'kCOV'] and R_in is not None:
            lda = self.calc_optimal_lambda(sigma_in)
            R_srk = self.correlation_shrinkage(R_in, lda=lda)
            sigma = np.matmul(np.matmul(Dhvar, R_srk), Dhvar)
        else:
            sigma = sigma_in

        # Full shrinkage if the above still produces a singular sigma matrix
        if np.linalg.det(sigma) == 0:
            print("Warning the covariance matrix is still Singular, proceeding to maximal shrinkage with lambda 1")
            R_maxsrk = self.correlation_shrinkage(R_in, lda=1)
            sigma = np.matmul(np.matmul(Dhvar, R_maxsrk), Dhvar)

        # Back to identity matrix if problem
        if np.linalg.det(sigma) == 0:
            print("Warning the covariance matrix is Singular by design after shrinkage")
            sigma = np.eye(self.output_length)

        return sigma

    def calc_optimal_lambda(self, sigma_in):
        """Computes the optimal estimated shrinkage intensity from a given covariance (sigma) input matrix"""
        cov_sigma = np.cov(sigma_in, bias=True)
        # Forcing sigma & cov_sigma diagonal elements to zero
        for i in range(np.shape(sigma_in)[0]):
            sigma_in[i, i] = 0
            cov_sigma[i, i] = 0
        sigma_elem_sum = np.sum(sigma_in ** 2)
        cov_sigma_elem_sum = np.sum(cov_sigma)
        try:
            # Obtaining optimal lambda coefficient
            optimal_lambda = cov_sigma_elem_sum / sigma_elem_sum
        except ZeroDivisionError:
            optimal_lambda = 1
        if optimal_lambda > 1 or optimal_lambda < 0:
            optimal_lambda = 1
        return optimal_lambda

    def correlation_shrinkage(self, corr_in, lda=None):
        """Correlation matrix shrinkage"""
        lda = 0.5 if lda is None else lda
        corrshrink = lda * np.eye(self.tree.n) + (1 - lda) * corr_in
        return corrshrink


#######################################损失函数#########################################

    def coherency_loss_function_mse(self, y_true, y_hat,scaler):
        # Accuracy loss - mean squared Error (MSE)
        accuracy_loss_mse = torch.mean(torch.square(y_true - y_hat))
        y_hat_x = scaler.inverse_transform(y_hat)
        # print('y_hat_x.dtype',y_hat_x.dtype)   32
        # print('self.SG_T.dtype',self.SG_T.dtype)   64
        self.SG_T = self.SG_T.to(torch.float32).to(device)
        SGyhat_unscaler = matrix_tensor_multiplication(self.SG_T, y_hat_x).to(device)
        SGyhat = scaler.transform(SGyhat_unscaler).to(device)
        coherency_loss = y_hat - SGyhat
        coherency_loss_mse = torch.mean(torch.square(coherency_loss))
        loss = accuracy_loss_mse * self.alpha + coherency_loss_mse * (1 - self.alpha)
        return loss

    def independent_loss_function_mse(self, y_true, y_hat, scaler=None):
        # Accuracy loss - Mean Squared squared Error (MSE)
        accuracy_loss_mse = torch.mean(torch.square(y_true - y_hat))
        return accuracy_loss_mse

    def coherency_loss_function_Q50(self,y_true,y_hat,scaler,quantiles):

        accuracy_loss_Q50 = quantileLoss(y_true,y_hat,quantiles=quantiles)
        y_hat_x = scaler.inverse_transform(y_hat[0])
        self.SG_T = self.SG_T.to(torch.float32).to(device)
        SGyhat_unscaler = matrix_tensor_multiplication(self.SG_T, y_hat_x).to(device)
        SGyhat = scaler.transform(SGyhat_unscaler).to(device)
        coherency_loss = y_hat[0] - SGyhat
        coherency_loss_Q50 = torch.mean(torch.square(coherency_loss))
        # loss = accuracy_loss_Q50 * learn_alpha + coherency_loss_Q50 * (1 - learn_alpha)
        return accuracy_loss_Q50,coherency_loss_Q50


#   y_hat是一个列表
    def coherency_loss_function_Q05_Q95(self, y_true, y_hat, scaler,median_forecast,
                                        quantiles):

        accuracy_loss_Q05_Q95 =quantileLoss(y_true, y_hat, quantiles=quantiles)
        y_hat_x_Q05 = scaler.inverse_transform(y_hat[0])
        y_hat_x_Q95 = scaler.inverse_transform(y_hat[1])
        self.SG_T = self.SG_T.to(torch.float32).to(device)
        self.SG_Q05_T = self.SG_Q05_T.to(torch.float32).to(device)
        self.SG_Q95_T = self.SG_Q95_T.to(torch.float32).to(device)
        SGyhat_unscaler_Q05 = matrix_tensor_multiplication(self.SG_Q05_T, y_hat_x_Q05).to(device)
        SGyhat_unscaler_Q95 = matrix_tensor_multiplication(self.SG_Q95_T, y_hat_x_Q95).to(device)
        SGyhat_scaler_Q05 = scaler.transform(SGyhat_unscaler_Q05).to(device)
        SGyhat_scaler_Q95 = scaler.transform(SGyhat_unscaler_Q95).to(device)
        #print(type(median_forecast), median_forecast.shape)
        coherency_loss_Q05 = torch.square(torch.square(y_hat[0] - median_forecast) - torch.square(SGyhat_scaler_Q05 - median_forecast))
        coherency_loss_Q95 = torch.square(torch.square(y_hat[1] - median_forecast) - torch.square(SGyhat_scaler_Q95 - median_forecast))

        coherency_loss_Q05_Q95 =torch.mean(coherency_loss_Q05+coherency_loss_Q95)

        return accuracy_loss_Q05_Q95,coherency_loss_Q05_Q95

    def QuantileLoss_Q50(self,y_true,y_hat,scaler,quantiles):

        accuracy_loss_Q50, coherency_loss_Q50 = self.coherency_loss_function_Q50(y_true, y_hat, scaler,quantiles)
        self.alpha_learn_Q_1 = torch.sigmoid(self.alpha_learn_Q)
        #print("11",accuracy_loss_Q50,"12",coherency_loss_Q50)
        loss = self.alpha_learn_Q_1 * accuracy_loss_Q50 + (1 - self.alpha_learn_Q_1) * coherency_loss_Q50
        #print(loss)
        return loss

    def QuantileLoss_Q05_Q95(self, y_true, y_hat, scaler, median_forecast, quantiles):
        accuracy_loss_Q05_095, coherency_loss_Q05_095  = self.coherency_loss_function_Q05_Q95(y_true,y_hat,scaler,median_forecast,quantiles)
        self.alpha_learn_Q_2 = torch.sigmoid(self.alpha_learn_Q)
        loss = self.alpha_learn_Q_2 * accuracy_loss_Q05_095 + (1 - self.alpha_learn_Q_2) * coherency_loss_Q05_095
        return loss

    def sharq_Q50(self,y_true, y_hat, scaler, quantiles, epsilon=.1):
        accuracy_loss_Q50 = quantileLoss(y_true, y_hat, quantiles=quantiles)
        y_hat_x = scaler.inverse_transform(y_hat[0])
        self.SG_T = self.SG_T.to(torch.float32).to(device)
        SGyhat_unscaler = matrix_tensor_multiplication(self.SG_T, y_hat_x).to(device)
        SGyhat = scaler.transform(SGyhat_unscaler).to(device)
        reg = y_hat[0] - SGyhat
        coherency_loss_Q50 = epsilon * torch.pow(reg, 2)
        loss = torch.mean(accuracy_loss_Q50+coherency_loss_Q50)
        return loss

    def sharq_accuracy_Q05_Q95(self, y_true, y_hat,quantiles):
        accuracy_loss_Q05_095 = quantileLoss(y_true, y_hat,quantiles=quantiles)
        return accuracy_loss_Q05_095

    def sharq_Q05_Q95(self,y_hat, scaler, median_forecast):
        y_hat_x_Q05 = scaler.inverse_transform(y_hat[0])
        y_hat_x_Q95 = scaler.inverse_transform(y_hat[1])
        self.SG_T = self.SG_T.to(torch.float32).to(device)
        self.SG_Q05_T = self.SG_Q05_T.to(torch.float32).to(device)
        self.SG_Q95_T = self.SG_Q95_T.to(torch.float32).to(device)
        SGyhat_unscaler_Q05 = matrix_tensor_multiplication(self.SG_Q05_T, y_hat_x_Q05).to(device)
        SGyhat_unscaler_Q95 = matrix_tensor_multiplication(self.SG_Q95_T, y_hat_x_Q95).to(device)
        SGyhat_scaler_Q05 = scaler.transform(SGyhat_unscaler_Q05).to(device)
        SGyhat_scaler_Q95 = scaler.transform(SGyhat_unscaler_Q95).to(device)
        coherency_loss_Q05 = torch.square(
            torch.square(y_hat[0] - median_forecast) - torch.square(SGyhat_scaler_Q05 - median_forecast))
        coherency_loss_Q95 = torch.square(
            torch.square(y_hat[1] - median_forecast) - torch.square(SGyhat_scaler_Q95 - median_forecast))
        loss = torch.mean(coherency_loss_Q05 + coherency_loss_Q95)
        return loss



"""
class QuantileLoss_Q50(nn.Module):
    def __init__(self, alpha_init=0.5):
        super(QuantileLoss_Q50, self).__init__()

        self.alpha = nn.Parameter(torch.tensor(alpha_init).clamp(0, 1))

    def forward(self, y_true,y_hat,scaler,quantiles):

        accuracy_loss_Q50, coherency_loss_Q50  = H_Regressor.coherency_loss_function_Q50(y_true,y_hat,scaler,quantiles)
        loss = self.alpha * accuracy_loss_Q50 + (1 - self.alpha) * coherency_loss_Q50
        return loss

class QuantileLoss_Q05_Q95(nn.Module):
    def __init__(self, alpha_init=0.5):
        super(QuantileLoss_Q05_Q95, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init).clamp(0, 1))

    def forward(self, y_true,y_hat,scaler,median_forecast,quantiles):
        accuracy_loss_Q05_095, coherency_loss_Q05_095  = H_Regressor.coherency_loss_function_Q05_Q95(y_true,y_hat,scaler,median_forecast,quantiles)
        loss = self.alpha * accuracy_loss_Q05_095 + (1 - self.alpha) * coherency_loss_Q05_095
        return loss

"""





"""
class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert len(preds) == len(self.quantiles)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[i]
            loss = torch.max((q - 1) * errors, q * errors).mean()
            losses.append(loss)
        return sum(losses)
"""




"""
class CoherencyLossFunctionMSE(nn.Module):
    def __init__(self, h_regressor):
        super(CoherencyLossFunctionMSE, self).__init__()
        self.h_regressor = h_regressor

    def forward(self, y_true, y_hat, scaler):
        return self.h_regressor.coherency_loss_function_mse(y_true, y_hat, scaler)

class IndependentLossFunctionMSE(nn.Module):
    def __init__(self, h_regressor):
        super(IndependentLossFunctionMSE, self).__init__()
        self.h_regressor = h_regressor

    def forward(self, y_true, y_hat, scaler):
        return self.h_regressor.independent_loss_function_mse(y_true, y_hat)
"""




