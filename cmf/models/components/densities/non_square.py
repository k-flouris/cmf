import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.autograd.functional as autograd_F

import numpy as np
import gc

sys.path.insert(0, str(Path(__file__).parents[4] / "gitmodules" / "gpytorch"))
try:
    from gpytorch.utils import linear_cg
finally:
    sys.path.pop(0)

from .density import Density
from .exact import BijectionDensity
from .split import SplitDensity


class NonSquareHeadDensity(Density):
    _VALID_LOG_JACOBIAN_METHODS = ["cholesky", "hutch_with_cg"]

    def __init__(
            self,
            prior,
            regularization_param,
            log_jacobian_method,
            x_shape,
            hutchinson_distribution,
            num_hutchinson_samples=1,
            max_cg_iterations=None,
            cg_tolerance=1,
            
            
):
        super().__init__()

        self.prior = prior
        self.regularization_param = regularization_param
        self.x_shape = x_shape

        self.hutchinson_distribution = hutchinson_distribution
        self.num_hutchinson_samples = num_hutchinson_samples
        self.max_cg_iterations = max_cg_iterations
        self.cg_tolerance = cg_tolerance
        if log_jacobian_method not in self._VALID_LOG_JACOBIAN_METHODS:
            raise ValueError(f"{log_jacobian_method} not a valid Jacobian calculation method")
        self.log_jacobian_method = log_jacobian_method

    def _ood(self, x):
        return self("elbo", x, ood=True)

    def _extract_latent(self, x, **kwargs):
        prior_dict = self.prior.elbo(x)
        low_dim_latent, _, earliest_latent = self._traverse_backward(x, prior_dict)

        if kwargs["earliest_latent"]:
            return earliest_latent
        else:
            return low_dim_latent

    def _elbo(self, x, add_reconstruction=True, add_diagonal_metric_reg=False, add_offdiagonal_metric_reg=False, likelihood_wt=1., metric_wt=1., visualization=False, ood=False, test_metric=False):
        prior_dict = self.prior.elbo(x)
        low_dim_latent, low_dim_elbo, _ = self._traverse_backward(x, prior_dict)

        if ood:
            assert self.log_jacobian_method == "cholesky"

        if not np.isclose(likelihood_wt, 0.):
            # NOTE: Combine log det jacobian and reconstruction because jvp will
            #       return reconstruction
            outputs = self._get_log_det_jac_and_reconstruction(
                latent=low_dim_latent,
                visualization=(visualization or ood)
            )
            if len(outputs) >= 2:
                log_det_jacobian, reconstructed_x, *rest = outputs
                jtj = rest[0] if len(rest) > 0 else None
            else:
                raise ValueError("outputs should contain at least two elements")

            
            likelihood_term = low_dim_elbo - log_det_jacobian/2.
            
            if add_diagonal_metric_reg:
                # NB. reconstructed from jvp_forawrd and reconstructed_x are the same
                # NB. using an autograd function does not work even if values are very similart to jvp full jac
                g_kk=torch.diagonal(jtj,  dim1=-2, dim2=-1)
                metric_l1_loss=torch.linalg.vector_norm(g_kk, ord=1, dim=1).unsqueeze(dim=-1) 
                del g_kk, jtj


            elif add_offdiagonal_metric_reg:
                # for 1D latent there is just diagonal anyway..
                # the view(BATCH,D(D+1)) i.e. SUM(D-1)*2 like factorial i.e. D(D-1)/2 *2
                g_i_not_j=jtj.masked_select(~torch.eye(jtj.shape[1],  device=jtj.device, dtype=bool)).view(jtj.shape[0],jtj.shape[1]*(jtj.shape[1]-1))
                metric_l1_loss=torch.linalg.vector_norm(g_i_not_j, ord=1, dim=1).unsqueeze(dim=-1) 
                del g_i_not_j, jtj    
                               
            else:
                metric_l1_loss = 0

        else:
            # NB. we also refrain from using l1 loss when likelihood is warming up
            metric_l1_loss = 0
            likelihood_term = 0            
            reconstructed_x = self.flow_forward(low_dim_latent)
            
        if add_reconstruction:
            assert not visualization
            reconstruction_errros = (reconstructed_x - x).flatten(start_dim=1)**2
            reconstruction_loss = torch.sum(reconstruction_errros, dim=-1, keepdim=True)
                        
        else:
            reconstruction_loss = 0

        if ood:
            assert add_reconstruction and not np.isclose(likelihood_wt, 0.)
            return {
                "likelihood": likelihood_term,
                "reconstruction-error": reconstruction_loss
            }

        return {
            "elbo": likelihood_wt*likelihood_term - self.regularization_param*reconstruction_loss - metric_wt*metric_l1_loss,
            "prior-dict": prior_dict
        }

    def _get_log_det_jac_and_reconstruction(self, latent, visualization):
        # NOTE: We will always run full Jacobians for testing, IF we get test elbo at all
        if not self.training or self.log_jacobian_method == "cholesky":
            log_det_jacobian_function = self._exact_log_det_jac_and_reconstruction
        elif self.log_jacobian_method == "hutch_with_cg":
            log_det_jacobian_function = self._approx_log_det_jac_and_reconstruction

        return log_det_jacobian_function(latent)

    def _sample(self, num_samples):
        return self.prior.sample(num_samples)

    def _fixed_sample(self, noise):
        return self.prior.fixed_sample(noise=noise)

    def _traverse_backward(self, x, prior_dict):
        """
        This function traverses backward through the transformations defining the flow.
        It outputs the low-dim latent variable and its log likelihood.
        It also modifies self.transform_stack and self.jvp_stack for self.flow_forward and
        self.jvp_forward, respectively.
        """
        transform_stack = []
        jvp_stack = []
        prior_pointer = self.prior

        while "low-dim-x" not in prior_dict:
            prior_dict = prior_dict["prior-dict"]
            jvp_stack.append(prior_pointer.jvp)

            if issubclass(type(prior_pointer), BijectionDensity):
                transform = prior_pointer.bijection.z_to_x
                prior_pointer = prior_pointer.prior
            elif issubclass(type(prior_pointer), SplitDensity):
                transform = prior_pointer.pad_inputs
                prior_pointer = prior_pointer.density_1
            else:
                raise ValueError(f"Cannot handle density of class {type(prior_pointer).__name__}")

            transform_stack.append(transform)

        jvp_stack.append(prior_pointer.jvp)
        transform_stack.append(prior_pointer.low_dim_to_masked)
        self._set_flow_and_jvp_stacks(transform_stack, jvp_stack)

        low_dim_latent = prior_dict["low-dim-x"]
        low_dim_elbo = prior_dict["elbo"]

        try:
            earliest_latent = prior_pointer.extract_latent(low_dim_latent)
        except NotImplementedError:
            earliest_latent = ""

        return low_dim_latent, low_dim_elbo, earliest_latent

    def _set_flow_and_jvp_stacks(self, transform_stack, jvp_stack):
        self.transform_stack = transform_stack[:]
        self.jvp_stack = jvp_stack[:]

    def _jac_transpose_jac_vec(self, latent, vec, create_graph):
        if not create_graph:
            latent = latent.detach().requires_grad_(False)
            with torch.no_grad():
                reconstruction, jvp = self.jvp_forward(latent, vec)
        else:
            reconstruction, jvp = self.jvp_forward(latent, vec)

        flow_forward_flat = lambda x: self.flow_forward(x).flatten(start_dim=1)
        _, jtjvp = autograd_F.vjp(flow_forward_flat, latent, jvp.flatten(start_dim=1), create_graph=create_graph)

        return jtjvp, reconstruction

    def _approx_log_det_jac_and_reconstruction(self, latent):
        sample_shape = (*latent.shape, self.num_hutchinson_samples)

        if self.hutchinson_distribution == "normal":
            hutchinson_samples = torch.randn(*sample_shape, device=latent.device)
        elif self.hutchinson_distribution == "rademacher":
            bernoulli_probs = 0.5*torch.ones(*sample_shape, device=latent.device)
            hutchinson_samples = torch.bernoulli(bernoulli_probs)
            hutchinson_samples.mul_(2.).subtract_(1.)
        else:
            raise ValueError(f"Unknown hutchinson distribution {self.hutchinson_distribution}")

        repeated_latent = latent.repeat_interleave(self.num_hutchinson_samples, dim=0)

        def tensor_to_vector(tensor):
            # Turn a tensor of shape (batch_size x latent_dim x num_hutch_samples)
            # into a vector of shape (batch_size*num_hutch_samples x latent_dim)
            # NOTE: Need to transpose first to get desired stacking from reshape
            vector = tensor.transpose(1,2).reshape(
                latent.shape[0]*self.num_hutchinson_samples, latent.shape[1]
            )
            return vector

        def vector_to_tensor(vector):
            # Inverse of `tensor_to_vector` above
            # NOTE: Again need to transpose to correctly unfurl num_hutch_samples as the final dimension
            tensor = vector.reshape(latent.shape[0], self.num_hutchinson_samples, latent.shape[1])
            return tensor.transpose(1,2)

        def jac_transpose_jac_closure(tensor):
            # NOTE: The CG method available to us expects a method to multiply against
            #       tensors of shape (batch_size x latent_dim x num_hutch_samples).
            #       Thus we need to wrap reshaping around our JtJv method,
            #       which expects v to be of shape (batch_size*num_hutch_samples x latent_dim).
            vec = tensor_to_vector(tensor)
            jtjvp, _ = self._jac_transpose_jac_vec(repeated_latent, vec, create_graph=False)
            return vector_to_tensor(jtjvp)

        jtj_inverse_hutchinson = linear_cg(
            jac_transpose_jac_closure,
            hutchinson_samples,
            max_iter=self.max_cg_iterations,
            max_tridiag_iter=self.max_cg_iterations,
            tolerance=self.cg_tolerance
        ).detach()

        jtj_hutchinson_vec, reconstruction_repeated = self._jac_transpose_jac_vec(
            repeated_latent, tensor_to_vector(hutchinson_samples), create_graph=self.training
        )
        reconstruction = reconstruction_repeated[::self.num_hutchinson_samples]
        jtj_hutchinson = vector_to_tensor(jtj_hutchinson_vec)

        # NOTE: jtj_inverse does not just cancel out with jtj because the former has a stop gradient applied.
        approx_log_det_jac = torch.mean(torch.sum(jtj_inverse_hutchinson*jtj_hutchinson, dim=1, keepdim=True), dim=2)

        return approx_log_det_jac, reconstruction, jtj_hutchinson # kf the last is used for l1 or whatever so its hard coded to be returned always



    def _exact_log_det_jac_and_reconstruction(self, latent):
        eps = 1e-6
        EPS_FACTOR = 10
        MAX_ATTEMPTS = 6

        batch_size = latent.shape[0]
        latent_dim = latent.shape[1]

        jacobian_t_jacobian, reconstructed_x = self._get_full_jac_transpose_jac(latent, self.training)

        jac_attempts = 1
        eye = torch.eye(
            latent_dim,
            device=jacobian_t_jacobian.device,
            dtype=jacobian_t_jacobian.dtype
            ).view(1, latent_dim, latent_dim)
        eye_repeat = eye.repeat((batch_size, 1, 1))

        while True:
            try:
                cholesky_factor = torch.linalg.cholesky(jacobian_t_jacobian)
                break
            except RuntimeError:
                # HACK: If we end up running into non-invertibility, add eps*I to JtJ
                jacobian_t_jacobian = jacobian_t_jacobian + eps*eye_repeat
                jac_attempts += 1
                eps *= EPS_FACTOR

        if jac_attempts > 1:
            print(f"WARNING: Numerical non-invertibility in JtJ observed - {jac_attempts} attempts needed to fix")

        cholesky_diagonal = torch.diagonal(cholesky_factor, dim1=-2, dim2=-1)
        log_det_jacobian = 2 * torch.sum(torch.log(cholesky_diagonal), dim=1, keepdim=True)

        return log_det_jacobian, reconstructed_x, jacobian_t_jacobian # kf the last is used for l1 or whatever so its hard coded to be returned always

    def _get_full_jac_transpose_jac(self, latent, create_graph):
        batch_size, matrix_dim = latent.shape

        jac = []
        for i in range(matrix_dim):
            vec = torch.zeros_like(latent)
            vec[:,i] = 1
            reconstruction, j_v = self.jvp_forward(latent, vec)
            jac.append(j_v.flatten(start_dim=1))
        jac = torch.stack(jac, dim=2)
        jac_transpose_jac = torch.bmm(jac.transpose(1, 2), jac)

        # Jacobian should end up as shape (batch_size, matrix_dim, matrix_dim)
        return jac_transpose_jac, reconstruction

    def flow_forward(self, x):
        # NOTE: Need to copy so that transform_stack is not cleared on repeated application
        transform_stack_copy = self.transform_stack[:]

        while transform_stack_copy:
            transform = transform_stack_copy.pop()
            x = transform(x)["x"]
        return x

    def jvp_forward(self, x, v):
        jvp_stack_copy = self.jvp_stack[:]

        while jvp_stack_copy:
            jvp_fn = jvp_stack_copy.pop()
            jvp_out = jvp_fn(x, v)
            x, v = jvp_out["x"], jvp_out["jvp"]
        return x, v

    def pullback_log_jac_jac_transpose(self, x):
        prior_dict = self.prior.elbo(x)
        low_dim_latent, _, _ = self._traverse_backward(x, prior_dict)

        jac = torch.autograd.grad(low_dim_latent, x, grad_outputs=torch.ones_like(low_dim_latent))[0]
        jac_jac_t = torch.sum(jac*jac, dim=1)

        return torch.log(jac_jac_t)


class ManifoldFlowHeadDensity(NonSquareHeadDensity):
    def _get_log_det_jac_and_reconstruction(self, latent, visualization):
        if visualization:
            return super()._get_log_det_jac_and_reconstruction(latent, visualization)
        else:
            return 0, self.flow_forward(latent)

    def separate_parameters(self, recurse=True):
        all_params = set(super().parameters())

        non_square_tail_density = self.prior
        while not isinstance(non_square_tail_density, NonSquareTailDensity):
            if issubclass(type(non_square_tail_density), BijectionDensity):
                non_square_tail_density = non_square_tail_density.prior
            elif issubclass(type(non_square_tail_density), SplitDensity):
                non_square_tail_density = non_square_tail_density.density_1

        likelihood_params = set(non_square_tail_density.parameters())
        reconstruction_params = all_params.difference(likelihood_params)

        reconstruction_params_generator = (p for p in reconstruction_params)
        likelihood_params_generator = (p for p in likelihood_params)

        return [reconstruction_params_generator, likelihood_params_generator]


class NonSquareTailDensity(Density):
    def __init__(self, prior, x_shape, latent_dimension, detach_before_prior):
        super().__init__()
        self.prior = prior
        self.detach_before_prior = detach_before_prior

        self.x_shape = x_shape
        self.latent_dimension = latent_dimension
        self.flattened_dims = np.prod(x_shape)

        self.register_buffer("mask", torch.arange(self.flattened_dims) < latent_dimension)
        self.register_buffer("permutation", torch.randperm(self.flattened_dims))
        self.register_buffer("inverse_permutation", torch.argsort(self.permutation))

    def _elbo(self, x):
        flattened_x = x.flatten(start_dim=1)
        permuted_x = flattened_x[:, self.permutation]
        low_dim_x = permuted_x[:, self.mask]

        if self.detach_before_prior:
            prior_dict = self.prior.elbo(low_dim_x.detach())
        else:
            prior_dict = self.prior.elbo(low_dim_x)

        return {
            "elbo": prior_dict["elbo"],
            "low-dim-x": low_dim_x,
            "prior-dict": prior_dict
        }

    def low_dim_to_masked(self, low_dim_x):
        batch_size = low_dim_x.shape[0]

        padded_x = torch.zeros((batch_size, self.flattened_dims)).to(low_dim_x.device)
        padded_x[:, :self.latent_dimension] = low_dim_x
        masked_x = padded_x[:, self.inverse_permutation].view((batch_size, *self.x_shape))

        return {"x": masked_x}

    def _jvp(self, x, v):
        return {
            "x": self.low_dim_to_masked(x)["x"],
            "jvp": self.low_dim_to_masked(v)["x"]
        }

    def _fixed_sample(self, noise):
        prior_sample = self.prior.fixed_sample(noise)
        return self.low_dim_to_masked(prior_sample)["x"]

    def _sample(self, num_samples):
        prior_sample = self.prior.sample(num_samples)
        return self.low_dim_to_masked(prior_sample)["x"]

    def _extract_latent(self, x, **kwargs):
        return self.prior.extract_latent(x, **kwargs)
    #
    # def _extract_metric(self, noise):
    #     return self.prior.extract_metric(noise)


# other version:
# class NonSquareHeadDensity(Density):
#     _VALID_LOG_JACOBIAN_METHODS = ["cholesky", "hutch_with_cg"]

#     def __init__(
#             self,
#             prior,
#             regularization_param,
#             log_jacobian_method,
#             x_shape,
#             hutchinson_distribution,
#             num_hutchinson_samples=1,
#             max_cg_iterations=None,
#             cg_tolerance=1,
#             #:
#             covariant_loss=False,
#             l1_regularized_loss=False,
#             metric_regularization_param=0.1
            
# ):
#         super().__init__()

#         self.prior = prior
#         self.regularization_param = regularization_param
#         self.x_shape = x_shape

#         # print("*****PRIOR******")
#         # for name, param in prior.named_parameters():
#         #     print(name, param.shape)
#         # print("****************")


#         self.hutchinson_distribution = hutchinson_distribution
#         self.num_hutchinson_samples = num_hutchinson_samples
#         self.max_cg_iterations = max_cg_iterations
#         self.cg_tolerance = cg_tolerance

#         if log_jacobian_method not in self._VALID_LOG_JACOBIAN_METHODS:
#             raise ValueError(f"{log_jacobian_method} not a valid Jacobian calculation method")
#         self.log_jacobian_method = log_jacobian_method
        
#         # 
#         self.covariant_loss=covariant_loss
#         self.metric_regularization_param = metric_regularization_param
#         self.l1_regularized_loss = l1_regularized_loss

#     def _ood(self, x):
#         return self("elbo", x, ood=True)

#     def _extract_latent(self, x, **kwargs):
#         prior_dict = self.prior.elbo(x)
#         low_dim_latent, _, earliest_latent = self._traverse_backward(x, prior_dict)

#         if kwargs["earliest_latent"]:
#             return earliest_latent
#         else:
#             return low_dim_latent

#     def _elbo(self, x, add_reconstruction=True, likelihood_wt=1., visualization=False, ood=False):
#         prior_dict = self.prior.elbo(x)
#         low_dim_latent, low_dim_elbo, _ = self._traverse_backward(x, prior_dict)
#         # if not add_reconstruction: print("********** no reco *****************")

#         if ood:
#             assert self.log_jacobian_method == "cholesky"

#         if not np.isclose(likelihood_wt, 0.):
#             # NOTE: Combine log det jacobian and reconstruction because jvp will
#             #       return reconstruction
#             # added the if here
#             if self.covariant_loss:
#              log_det_jacobian, reconstructed_x, jtj = self._get_log_det_jac_and_reconstruction(
#                  # kf can we pass the full latent here maybe to get the full jacobian?
#                  latent=low_dim_latent,
#                  visualization=(visualization or ood),
#                  )           
#             else:    
#                 log_det_jacobian, reconstructed_x = self._get_log_det_jac_and_reconstruction(
#                     latent=low_dim_latent,
#                     visualization=(visualization or ood),
#                 )
#             #  here log det jacobian must be J_theta but why 1x1? for latent dim 1? because of rectangularness
#             #  low dim elbow must contain the first 2 parts of likelihood term pz and j_h
#             likelihood_term = low_dim_elbo - log_det_jacobian/2.

#         else:
#             likelihood_term = 0
#             reconstructed_x = self.flow_forward(low_dim_latent)
#             #
#             jtj = None
            
#         if add_reconstruction:
#             assert not visualization

#             # added this if here
#             if self.covariant_loss and jtj is not None: 
#                 #### This is also wrong we need the total metric again to do this!
#                 # will need this to be different for higher dimensions?
#                 # ------ if not np.isclose bypasses all this? for both methods? maybe need some thing to enforce it? -----
#                 identity=torch.diag(torch.tensor((1,1))).repeat(reconstructed_x.shape[0],1,1)
#                 identity=identity.type(torch.FloatTensor) # can use torch.eye
#                 regularized_jtj=identity+self.metric_regularization_param*jtj
#                 ### correct einsum      einsum=torch.einsum("bki,bkj->bij",jacobian_outer,jacobian_outer)
#                 # kf identity needs to 2x2 which is a bit of a problem? calclulate different g? or find a way to calcualte with the 1x1? For 1D case!
#                 d= (reconstructed_x - x)
#                 gd=torch.einsum('bij,bj->bi', regularized_jtj,d)
#                 dgd=torch.einsum('bi,bi->b', d,gd)
#                 reconstruction_loss=dgd.unsqueeze(dim=1)
#                 #  kf debugging by checking with 
#                 # reconstruction_errros = (reconstructed_x - x).flatten(start_dim=1)**2
#                 # reconstruction_loss_test = torch.sum(reconstruction_errros, dim=-1, keepdim=True)
#                 # # This works as the squaring but with jtj=1 it doesnt 
#                 # d1d=torch.einsum('bi,bi->b', d,d)
#                 # d1d=d1d.unsqueeze(dim=1)
#                 # print(jtj.shape,d.shape,(reconstruction_loss-d1d).mean())
#                 # print("********** diffwerence ",(reconstruction_loss_test - reconstruction_loss).mean(), "*****************")
#                 # print("********** cov reconstruction_loss shape ",reconstruction_loss.shape, "*****************")
#                 # print("********** g ",jtj.shape, "*****************")
#             elif self.l1_regularized_loss: #and jtj is not None: NEEDS FIXING HERE THIS IS NOT SO SIMPLE NEED TO CHANGE THE ELBO OF EVERYTHING? maybe do this in the experiment?
#                 reconstruction_errros = (reconstructed_x - x).flatten(start_dim=1)**2
#                 reconstruction_loss = torch.sum(reconstruction_errros, dim=-1, keepdim=True)
#                 # reconstruction_loss -= self.metric_regularization_param*torch.exp(low_dim_elbo) #kf this is wrong... need to implement it differentely 
#             else:
#                 reconstruction_errros = (reconstructed_x - x).flatten(start_dim=1)**2
#                 reconstruction_loss = torch.sum(reconstruction_errros, dim=-1, keepdim=True)
#                 # print("********** reconstruction_loss shape ",reconstruction_loss.shape, "*****************")
#         else:
#             reconstruction_loss = 0

#         if ood:
#             assert add_reconstruction and not np.isclose(likelihood_wt, 0.)
#             return {
#                 "likelihood": likelihood_term,
#                 "reconstruction-error": reconstruction_loss
#             }

#         return {
#             "elbo": likelihood_wt*likelihood_term - self.regularization_param*reconstruction_loss,
#             "prior-dict": prior_dict
#         }
#     def _get_log_det_jac_and_reconstruction(self, latent, visualization):
#         # NOTE: We will always run full Jacobians for testing, IF we get test elbo at all
#         if not self.training or self.log_jacobian_method == "cholesky":
#             log_det_jacobian_function = self._exact_log_det_jac_and_reconstruction
#         elif self.log_jacobian_method == "hutch_with_cg":
#             log_det_jacobian_function = self._approx_log_det_jac_and_reconstruction

#         return log_det_jacobian_function(latent)

#     def _sample(self, num_samples):
#         return self.prior.sample(num_samples)

#     def _fixed_sample(self, noise):
#         return self.prior.fixed_sample(noise=noise)

#     def _traverse_backward(self, x, prior_dict):
#         """
#         This function traverses backward through the transformations defining the flow.
#         It outputs the low-dim latent variable and its log likelihood.
#         It also modifies self.transform_stack and self.jvp_stack for self.flow_forward and
#         self.jvp_forward, respectively.
#         """
#         transform_stack = []
#         jvp_stack = []
#         prior_pointer = self.prior

#         while "low-dim-x" not in prior_dict:
#             #KF print(prior_dict.keys(),prior_dict["prior-dict"].keys())
#             #KF the latter has a low-dim-x in it after the 5th block or similar
#             prior_dict = prior_dict["prior-dict"]
#             jvp_stack.append(prior_pointer.jvp)

#             if issubclass(type(prior_pointer), BijectionDensity):
#                 transform = prior_pointer.bijection.z_to_x
#                 prior_pointer = prior_pointer.prior
#             elif issubclass(type(prior_pointer), SplitDensity):
#                 transform = prior_pointer.pad_inputs
#                 prior_pointer = prior_pointer.density_1
#             else:
#                 raise ValueError(f"Cannot handle density of class {type(prior_pointer).__name__}")

#             transform_stack.append(transform)

#         jvp_stack.append(prior_pointer.jvp)
#         transform_stack.append(prior_pointer.low_dim_to_masked)
#         self._set_flow_and_jvp_stacks(transform_stack, jvp_stack)

#         low_dim_latent = prior_dict["low-dim-x"]
# #  kf this comes from factory bijection (eg) exact.py, elbo.py where it rurns prior-dict, elbo, bijection
#         low_dim_elbo = prior_dict["elbo"]


#         try:
#             earliest_latent = prior_pointer.extract_latent(low_dim_latent)
#         except NotImplementedError:
#             earliest_latent = ""
        

#         return low_dim_latent, low_dim_elbo, earliest_latent

#     def _set_flow_and_jvp_stacks(self, transform_stack, jvp_stack):
#         self.transform_stack = transform_stack[:]
#         self.jvp_stack = jvp_stack[:]

#     def _jac_transpose_jac_vec(self, latent, vec, create_graph):
#         if not create_graph:
#             latent = latent.detach().requires_grad_(False)
#             with torch.no_grad():
#                 reconstruction, jvp = self.jvp_forward(latent, vec)
#         else:
#             reconstruction, jvp = self.jvp_forward(latent, vec)

#         flow_forward_flat = lambda x: self.flow_forward(x).flatten(start_dim=1)
#         _, jtjvp = autograd_F.vjp(flow_forward_flat, latent, jvp.flatten(start_dim=1), create_graph=create_graph)

#         return jtjvp, reconstruction

#     def _approx_log_det_jac_and_reconstruction(self, latent):
#         sample_shape = (*latent.shape, self.num_hutchinson_samples)

#         if self.hutchinson_distribution == "normal":
#             hutchinson_samples = torch.randn(*sample_shape, device=latent.device)
#         elif self.hutchinson_distribution == "rademacher":
#             bernoulli_probs = 0.5*torch.ones(*sample_shape, device=latent.device)
#             hutchinson_samples = torch.bernoulli(bernoulli_probs)
#             hutchinson_samples.mul_(2.).subtract_(1.)
#         else:
#             raise ValueError(f"Unknown hutchinson distribution {self.hutchinson_distribution}")

#         repeated_latent = latent.repeat_interleave(self.num_hutchinson_samples, dim=0)

#         def tensor_to_vector(tensor):
#             # Turn a tensor of shape (batch_size x latent_dim x num_hutch_samples)
#             # into a vector of shape (batch_size*num_hutch_samples x latent_dim)
#             # NOTE: Need to transpose first to get desired stacking from reshape
#             vector = tensor.transpose(1,2).reshape(
#                 latent.shape[0]*self.num_hutchinson_samples, latent.shape[1]
#             )
#             return vector

#         def vector_to_tensor(vector):
#             # Inverse of `tensor_to_vector` above
#             # NOTE: Again need to transpose to correctly unfurl num_hutch_samples as the final dimension
#             tensor = vector.reshape(latent.shape[0], self.num_hutchinson_samples, latent.shape[1])
#             return tensor.transpose(1,2)

#         def jac_transpose_jac_closure(tensor):
#             # NOTE: The CG method available to us expects a method to multiply against
#             #       tensors of shape (batch_size x latent_dim x num_hutch_samples).
#             #       Thus we need to wrap reshaping around our JtJv method,
#             #       which expects v to be of shape (batch_size*num_hutch_samples x latent_dim).
#             vec = tensor_to_vector(tensor)
#             jtjvp, _ = self._jac_transpose_jac_vec(repeated_latent, vec, create_graph=False)
#             return vector_to_tensor(jtjvp)

#         jtj_inverse_hutchinson = linear_cg(
#             jac_transpose_jac_closure,
#             hutchinson_samples,
#             max_iter=self.max_cg_iterations,
#             max_tridiag_iter=self.max_cg_iterations,
#             tolerance=self.cg_tolerance
#         ).detach()

#         jtj_hutchinson_vec, reconstruction_repeated = self._jac_transpose_jac_vec(
#             repeated_latent, tensor_to_vector(hutchinson_samples), create_graph=self.training
#         )
#         reconstruction = reconstruction_repeated[::self.num_hutchinson_samples]
#         jtj_hutchinson = vector_to_tensor(jtj_hutchinson_vec)

#         # NOTE: jtj_inverse does not just cancel out with jtj because the former has a stop gradient applied.
#         # NOTE: Stop gradient is used for the J_theta full transformation
#         approx_log_det_jac = torch.mean(torch.sum(jtj_inverse_hutchinson*jtj_hutchinson, dim=1, keepdim=True), dim=2)

#         return approx_log_det_jac, reconstruction

# #   kf added return full
#     def _exact_log_det_jac_and_reconstruction(self, latent):
#         eps = 1e-6
#         EPS_FACTOR = 10
#         MAX_ATTEMPTS = 6

#         batch_size = latent.shape[0]
#         latent_dim = latent.shape[1]

#         jacobian_t_jacobian, reconstructed_x = self._get_full_jac_transpose_jac(latent, self.training)

#         jac_attempts = 1
#         eye = torch.eye(
#             latent_dim,
#             device=jacobian_t_jacobian.device,
#             dtype=jacobian_t_jacobian.dtype
#             ).view(1, latent_dim, latent_dim)
#         eye_repeat = eye.repeat((batch_size, 1, 1))

#         while True:
#             try:
#                 cholesky_factor = torch.linalg.cholesky(jacobian_t_jacobian)
#                 break
#             except RuntimeError:
#                 # HACK: If we end up running into non-invertibility, add eps*I to JtJ
#                 jacobian_t_jacobian = jacobian_t_jacobian + eps*eye_repeat
#                 jac_attempts += 1
#                 eps *= EPS_FACTOR

#         if jac_attempts > 1:
#             print(f"WARNING: Numerical non-invertibility in JtJ observed - {jac_attempts} attempts needed to fix")

#         cholesky_diagonal = torch.diagonal(cholesky_factor, dim1=-2, dim2=-1)
#         log_det_jacobian = 2 * torch.sum(torch.log(cholesky_diagonal), dim=1, keepdim=True)
        
#         # kf:
#         if self.covariant_loss:
#             return log_det_jacobian, reconstructed_x, jacobian_t_jacobian

#         return log_det_jacobian, reconstructed_x

#     def _get_full_jac_transpose_jac(self, latent, create_graph):
#         batch_size, matrix_dim = latent.shape
# #kf This is JthetaJtheta but has the lantent dimension/ i.e. the relevant one as it is rectagular!
# #kf how to use this jvp forwar to get full dimemension? maybe can pass everything instead of lower latent!
#         jac = []
#         for i in range(matrix_dim):
#             vec = torch.zeros_like(latent)
#             vec[:,i] = 1
#             reconstruction, j_v = self.jvp_forward(latent, vec)
#             jac.append(j_v.flatten(start_dim=1))
#         jac = torch.stack(jac, dim=2)
#         jac_transpose_jac = torch.bmm(jac.transpose(1, 2), jac)

#         # Jacobian should end up as shape (batch_size, matrix_dim, matrix_dim)
#         return jac_transpose_jac, reconstruction

#     def flow_forward(self, x):
#         # NOTE: Need to copy so that transform_stack is not cleared on repeated application
#         transform_stack_copy = self.transform_stack[:]

#         while transform_stack_copy:
#             transform = transform_stack_copy.pop()
#             x = transform(x)["x"]
#         return x

#     def jvp_forward(self, x, v):
#         jvp_stack_copy = self.jvp_stack[:]

#         while jvp_stack_copy:
#             jvp_fn = jvp_stack_copy.pop()
#             jvp_out = jvp_fn(x, v)
#             x, v = jvp_out["x"], jvp_out["jvp"]
#         return x, v

#     def pullback_log_jac_jac_transpose(self, x):
#         prior_dict = self.prior.elbo(x)
#         low_dim_latent, _, _ = self._traverse_backward(x, prior_dict)

#         jac = torch.autograd.grad(low_dim_latent, x, grad_outputs=torch.ones_like(low_dim_latent))[0]
#         jac_jac_t = torch.sum(jac*jac, dim=1)

#         return torch.log(jac_jac_t)


# class ManifoldFlowHeadDensity(NonSquareHeadDensity):
#     def _get_log_det_jac_and_reconstruction(self, latent, visualization):
#         if visualization:
#             return super()._get_log_det_jac_and_reconstruction(latent, visualization)
#         else:
#             return 0, self.flow_forward(latent)

#     def separate_parameters(self, recurse=True):
#         all_params = set(super().parameters())

#         non_square_tail_density = self.prior
#         while not isinstance(non_square_tail_density, NonSquareTailDensity):
#             if issubclass(type(non_square_tail_density), BijectionDensity):
#                 non_square_tail_density = non_square_tail_density.prior
#             elif issubclass(type(non_square_tail_density), SplitDensity):
#                 non_square_tail_density = non_square_tail_density.density_1

#         likelihood_params = set(non_square_tail_density.parameters())
#         reconstruction_params = all_params.difference(likelihood_params)

#         reconstruction_params_generator = (p for p in reconstruction_params)
#         likelihood_params_generator = (p for p in likelihood_params)

#         return [reconstruction_params_generator, likelihood_params_generator]
