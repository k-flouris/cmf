import os

from collections import defaultdict

import numpy as np

import torch
import torch.utils.data
import torchvision.utils

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (4,4)
from matplotlib.collections import LineCollection

import tqdm

from .metrics import metrics

from scipy.special import i0
from scipy.stats import vonmises
from scipy.stats import gaussian_kde
from sklearn import preprocessing

# matplotlib.use('Agg')

import random
import json
# import seaborn
try: from .kf_fid_score import fid_from_samples, m_s_from_samples
except: None
# TODO: Make return a matplotlib figure instead. Writing can be done outside.
limitaxes=False


class DensityVisualizer:
    def __init__(self, writer):
        self._writer = writer

    def visualize(self, density, epoch):
        raise NotImplementedError


class DummyDensityVisualizer(DensityVisualizer):
    def visualize(self, density, epoch):
        return


class ImageDensityVisualizer(DensityVisualizer):
    @torch.no_grad()
    def visualize(self, density, epoch, write_folder=None, fixed_noise=None, extent=None, labels=None):
        density.eval()
        imgs = density.fixed_sample(fixed_noise)

        num_images = imgs.shape[0]
        num_rows = int(np.ceil(num_images / min(np.sqrt(num_images), 10)))

        # NOTE: This may not normalize exactly as we would like, since we might not
        #       observe the full range of pixel values. It may be best to specify the
        #       range as the standard {0, ..., 255}, but just leaving it for now.
        grid = torchvision.utils.make_grid(
            imgs, nrow=num_rows, pad_value=1,
            normalize=True, scale_each=True
        )
        grid_permuted = grid.permute((1,2,0))
        plt.imshow(grid_permuted.detach().cpu().numpy(), extent=extent)
        if labels:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])

        if write_folder:
            plt.savefig(os.path.join(write_folder, "samples.pdf"))
        else:
            self._writer.write_image("samples", grid, global_step=epoch)


# ---------------------------------------------------------------------------------- 
class ImageMetricDensityVisualizer(DensityVisualizer):
    def __init__(self, writer, x_train, device, num_elbo_samples, test_loader=None):
        super().__init__(writer=writer)

        self._x = x_train
        self._device = device
        self._num_elbo_samples = num_elbo_samples
        self._test_loader = test_loader
        
    def set_seed(self, s=42):
        np.random.seed(s)
        random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(s)
        print(f"Random seed set as {s}")
    
    
    # @torch.no_grad()
    def visualize(self, density, epoch, write_folder=None, fixed_noise=None, extent=None, labels=None):
        # These are for testing and analysis with a trained model - uncomment acordigly 
        
        density.eval()
        metric_sort_index=self.calculate_metric_and_sort_ld(density)
        variance_sort_index=self.calculate_variance_and_sort_ld(density, write_folder)
        # self.plot_metric_components(metric_sort_index,density, epoch, write_folder, fixed_noise, extent, labels)

        # self.plot_fid_for_effective_z(variance_sort_index,density, epoch, write_folder, fixed_noise, extent, labels)
        # self.plot_pairwise_dimension_comparison(metric_sort_index,density, epoch, write_folder, fixed_noise, extent, labels)
        self.plot_samples_prominent_z_combined_manifold_plotting(variance_sort_index,density, epoch, write_folder, fixed_noise, extent, labels, num_of_dims=10,zrange=3)

        # self.plot_samples_prominent_z_indivitual(metric_sort_index,density, epoch, write_folder, fixed_noise, extent, labels, bs=15, num_of_dims=10)
        # self.plot_samples_prominent_z_combined(metric_sort_index,density, epoch, write_folder, fixed_noise, labels, extent=3,num_of_dims=5)
        # self.plot_samples_prominent_z_cumulative(metric_sort_index,density, epoch, write_folder, fixed_noise, extent, labels, bs=15,num_of_dims=10)
        
        # self.plot_samples_prominent_z_hierarchical(metric_sort_index,density, epoch, write_folder, fixed_noise, extent, labels, num_of_dims=10)
        
        # NEED FIXING:
        # self.plot_two_dim_manifold_for_largest_effective_d(metric_sort_index,density, epoch, write_folder)
        # self.plot_input_images(density, epoch, write_folder, fixed_noise, extent, labels)
        # self.plot_samples_for_effective_z(metric_sort_index,density, epoch, write_folder, fixed_noise, extent, labels)
        # self.plot_expectation_zero(metric_sort_index,density, epoch, write_folder, fixed_noise, extent, labels)
        # self.plot_individual_latent(density, epoch, write_folder, fixed_noise, extent, labels)
    
    # @torch.no_grad()
    def visualize_color(self, density, epoch, write_folder=None, fixed_noise=None, extent=None, labels=None):
        density.eval()
        # metric_sort_index=self.calculate_metric_and_sort_ld(density)
        # self.calculate_metric_and_sort_ld(density)
        self.plot_input_images(density, epoch, write_folder, fixed_noise, extent, labels)

        
        
    
 
    
    def plot_input_images(self, density, epoch, write_folder=None, fixed_noise=None, extent=None, labels=None):
        imgs=self._x[-10:]
        # print(_x.shape)
        num_images = imgs.shape[0]
        num_rows = int(np.ceil(num_images / min(np.sqrt(num_images), 10)))

        # NOTE: This may not normalize exactly as we would like, since we might not
        #       observe the full range of pixel values. It may be best to specify the
        #       range as the standard {0, ..., 255}, but just leaving it for now.
        grid = torchvision.utils.make_grid(
            imgs, nrow=num_rows, pad_value=1,
            normalize=True, scale_each=True
        )
        grid_permuted = grid.permute((1,2,0))
        plt.imshow(grid_permuted.detach().cpu().numpy(), extent=extent)
        if labels:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])

        # if write_folder:
        print(write_folder)
        if True:
            write_folder='runs/'
            savedir = os.path.join(write_folder, "test_input_images")
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            plt.savefig(os.path.join(savedir, "plot_input_images.pdf")) 
        else:
            self._writer.write_image("plot_input_iamges", grid, global_step=epoch)

    # @torch.no_grad()
    def plot_metric_components(self,sort_index, density, epoch, write_folder=None, fixed_noise=None, extent=None, labels=None, num_of_dims=5, bs=16):

        import scipy
        density.eval()
        x=self._x[-bs:] 

        # determine ld and reverse index:
        low_dim_latent=density.extract_latent(x,earliest_latent=True)
        low_dim_latent=low_dim_latent.to(self._device)
        ld=low_dim_latent.shape[1]
        reverse_sort_index = sorted(range(ld), key=sort_index.__getitem__)

        # ALTERNATIVE:
        # noise=torch.randn(1,ld).to(self._device)
        # low_dim_latent=noise
        
        mult=int(ld/num_of_dims)
        dimensions=list(range(num_of_dims+1))
        dimensions=[item * mult for item in dimensions]
        
        print("## calulating FID and recon for effective dimensions {}".format(dimensions))
        
        _x=self._x[-128:]
        _x.requires_grad=True
        low_dim_latent=density.extract_latent(_x,earliest_latent=True)
        # low_dim_latent=low_dim_latent.to(torch.device("cuda"), dtype=torch.float32)
        x_= density.fixed_sample(low_dim_latent) 
        dxdz = self.jacobian(x_,low_dim_latent)
        g=torch.einsum("bki,bkj->bij",dxdz,dxdz)
        g_kk=torch.diagonal(g,  dim1=-2, dim2=-1)

           
        savedir = os.path.join(write_folder, "test_metric/")
        if not os.path.exists(savedir):
            os.makedirs(savedir)
            
        v1=g_kk.mean(axis=0)
        v1 = v1 / torch.max(torch.abs(v1))
        
        cmap = plt.cm.RdBu
        fig, ax = plt.subplots(1, 2)
        v1 = np.reshape(v1, (1, v1.shape[0]))
        # im1 = ax.imshow(v1, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        
        v2=g.mean(axis=0)
        v2 = v2 / torch.max(torch.abs(v2))

        # v = np.reshape(v, (1, v.shape[0]))
        # im2 = ax.imshow(v2, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        # plt.colorbar(mappable=im2,orientation='vertical')
        
        
        fig, (ax1, ax2) = plt.subplots(2, 1)
        
        # Plot the first image in the top subplot
        im1=ax1.imshow(v1, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        fig.colorbar(im1, ax=ax1, orientation='horizontal')

        # Plot the second image in the bottom subplot
        im2=ax2.imshow(v2, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        fig.colorbar(im2, ax=ax2, orientation='vertical')
        # Adjust the aspect ratio of the top subplot
        ax1.set_aspect(1 / 4)
        
        # Remove axis ticks and labels from both subplots
        ax1.axis('off')
        ax2.axis('off')
        
        # Adjust the spacing between subplots
        plt.subplots_adjust(hspace=0.0)
        
        
        if write_folder:
            place=os.path.join(savedir, "plot_g.pdf")
            plt.savefig(place)
            print("wrote at {}".format(place))
            
        gTg = torch.einsum('nij,nik->njk', g, g)   
        vTv=gTg.mean(axis=0)
        vTv = vTv / torch.max(torch.abs(vTv))
        fig, ax = plt.subplots(1, 1)
        # v = np.reshape(v, (1, v.shape[0]))
        im = ax.imshow(vTv, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        plt.axis('off') 

        if write_folder:
            place=os.path.join(savedir, "plot_gTg.pdf")
            plt.savefig(place)
            print("wrote at {}".format(place))
            
        plt.clf()    
            
        U, S, V = torch.svd(g)

        # Compute the dot product of U with its transpose
        UU_T = torch.matmul(U, U.transpose(1, 2))
        
        # Compute the dot product of V with its transpose
        VV_T = torch.matmul(V, V.transpose(1, 2))
        
        # Calculate the orthogonality measures for each sample in the batch
        U_orthogonality = torch.norm(torch.eye(U.size(1)).unsqueeze(0).expand(U.size(0), -1, -1) - UU_T, dim=(1,2))
        V_orthogonality = torch.norm(torch.eye(V.size(1)).unsqueeze(0).expand(V.size(0), -1, -1) - VV_T, dim=(1,2))
        
        # Save the orthogonality measures in a JSON file
        orthogonality_dict = {"U_orthogonality": U_orthogonality.tolist(), "V_orthogonality": V_orthogonality.tolist()}
        with open(os.path.join(savedir,"orthogonality.json"), "w") as f:
            json.dump(orthogonality_dict, f)    

        # Compute the SVD for each tensor in the batch
        singular_values = []
        for i in range(g.shape[0]):
            U, S, V = torch.svd(g[i])
            singular_values.append(S)
        
        # Compute the average singular values across the batch
        average_singular_values = torch.mean(torch.stack(singular_values), dim=0)
        
        # Compute the "average orthogonality" as the mean of the reciprocal of the singular values
        average_orthogonality = torch.mean(1.0 / average_singular_values)
        
        # Save the result in a JSON file
        result = {'average_orthogonality': average_orthogonality.item()}
        with open(os.path.join(savedir,"average_orthogonality.json"), "w") as f:
            json.dump(result, f)        
        
        # ENTROPY:
        matrix_entropy=np.zeros(g.shape[0])    
        for i in range(g.shape[0]):
            matrix_entropy[i] = scipy.stats.entropy(g[i].flatten())
            
        avr_entropy=np.mean(matrix_entropy)  
        result = {'entropy': avr_entropy.item()}
        with open(os.path.join(savedir,"entropy.json"), "w") as f:
            json.dump(result, f)  
            
        # norms = np.linalg.norm(g, axis=-1, keepdims=True)
        # cosine_similarity = np.einsum('bij,bkj->bik', g, g) / (norms * norms.transpose(0, 2, 1))
        # Calculate the norm of dxdz
        norm_dxdz = torch.norm(dxdz, dim=2, keepdim=True)
        
        # Normalize dxdz
        normalized_dxdz = dxdz / (norm_dxdz + 1e-8)
        
        # Calculate the cosine similarity
        cosine_similarity = torch.einsum("bki,bkj->bij", normalized_dxdz, normalized_dxdz)
        
        # Calculate the average cosine similarity over the batch
        average_cosine_similarity = torch.mean(cosine_similarity, dim=0)        # Average the cosine similarity across the batch
                # average_cosine_similarity = np.mean(cosine_similarity, axis=0)
        print("SHAPE",average_cosine_similarity.shape)
        # Reshape to create a single 2D image plot
        average_cosine_similarity = average_cosine_similarity.reshape(1, average_cosine_similarity.shape[1], average_cosine_similarity.shape[1])
                # Plot the average cosine similarity matrix
        im=plt.imshow(average_cosine_similarity[0],vmin=-0.3, vmax=0.3, cmap=cmap)
        plt.colorbar(mappable=im)     
        plt.axis('off') 

        if write_folder:
            place=os.path.join(savedir, "plot_cosine_similarity.pdf")
            plt.savefig(place)
            print("wrote at {}".format(place))
        
        
        
        v1=g_kk.mean(axis=0)
        v1 = v1 / torch.max(torch.abs(v1))
        
        cmap = plt.cm.RdBu
        fig, ax = plt.subplots(1, 2)
        v1 = np.reshape(v1, (1, v1.shape[0]))
        # im1 = ax.imshow(v1, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        

        # v = np.reshape(v, (1, v.shape[0]))
        # im2 = ax.imshow(v2, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        # plt.colorbar(mappable=im2,orientation='vertical')
        
        
        fig, (ax1, ax2) = plt.subplots(2, 1,figsize=plt.figaspect(2))

        # Plot the first image in the top subplot
        im1=ax1.imshow(v1, cmap=cmap, vmin=-1, vmax=1)
        cbar=fig.colorbar(im1, ax=ax1, orientation='horizontal')
        cbar.set_label(r"$G_{kk}$")

        # Plot the second image in the bottom subplot
        im2=ax2.imshow(average_cosine_similarity[0], cmap=cmap, vmin=-0.1, vmax=0.1, aspect='auto')
        cbar = fig.colorbar(im2, ax=ax2, orientation='vertical')
        cbar.set_label("Cosine similarity")
        # Adjust the aspect ratio of the top subplot
        ax1.set_aspect(0.6)

        # Remove axis ticks and labels from both subplots
        ax1.axis('off')
        ax2.axis('off')
        
        mean_abs_value = torch.mean(torch.abs(average_cosine_similarity[0]))
        # Adjust the spacing between subplots
        plt.subplots_adjust(hspace=0.1)
        
        result = {'mean absolut cosine similarity': mean_abs_value.item()}
        with open(os.path.join(savedir,"mean_absolut_cosine_similarity.json"), "w") as f:
            json.dump(result, f)  
            
            
        if write_folder:
            place=os.path.join(savedir, "plot_g_combined_01.pdf")
            plt.savefig(place, bbox_inches='tight')
            
            print("wrote at {}".format(place))
        
    def calculate_metric_and_sort_ld(self, density):
        _x=self._x[-64:]
        # print(_x.shape)
        _x.requires_grad=True
        low_dim_latent=density.extract_latent(_x,earliest_latent=True)
        # low_dim_latent=low_dim_latent.to(torch.device("cuda"), dtype=torch.float32)
        x_= density.fixed_sample(low_dim_latent) 
        dxdz = self.jacobian(x_,low_dim_latent)
        g=torch.einsum("bki,bkj->bij",dxdz,dxdz)
        g_kk=torch.diagonal(g,  dim1=-2, dim2=-1)
    
        # Maybe sensitive cuda for parallelization
        # jtj, _ = density._get_full_jac_transpose_jac(low_dim_latent, False)
        # log_jac_jac_t = density.pullback_log_jac_jac_transpose(low_dim_latent)
        g_kk_sort_index = np.argsort(torch.abs(g_kk.mean(axis=0)))

        return g_kk_sort_index
    
    def calculate_variance_and_sort_ld(self, density, write_folder=None):
        _x=self._x[-128:]
        # print(_x.shape)
        _x.requires_grad=True
        low_dim_latent=density.extract_latent(_x,earliest_latent=True)
        print(low_dim_latent.shape)
        # low_dim_latent=low_dim_latent.to(torch.device("cuda"), dtype=torch.float32)
        variances_z = torch.var(low_dim_latent, dim=0, unbiased=False)  # dim=0 calculates variance across the batch dimension
        variances_z=variances_z.cpu().detach().numpy()

        var_sort_index = np.argsort(np.abs(variances_z))
        sorted_variances_z_desc = variances_z[var_sort_index[::-1]]

        x_= density.fixed_sample(low_dim_latent) 

        dimensions=np.arange(len(sorted_variances_z_desc))
        print(variances_z)
    
        # Maybe sensitive cuda for parallelization
        # jtj, _ = density._get_full_jac_transpose_jac(low_dim_latent, False)
        # log_jac_jac_t = density.pullback_log_jac_jac_transpose(low_dim_latent)

        zvars_dict = dict(zip(dimensions,sorted_variances_z_desc))
        zvars_dict = {int(k): v.item() for k, v in zvars_dict.items()}

        # print("zVARs:", zvars_dict)
        savedir = os.path.join(write_folder, "test_metric/")
        writepath = os.path.join(savedir, "zvars.json")
        json_dump = json.dumps(zvars_dict, indent=4)
        
        with open(writepath, "w") as f:
            f.write(json_dump)

        fig, ax = plt.subplots(1, 1)

        plt.plot(dimensions,sorted_variances_z_desc, 'k.')
        ax.set_xlabel(r'latent dimension', fontsize=15)
        ax.set_ylabel(r'variance', fontsize=15)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc=1, fontsize=7)
        plt.title
        plt.tight_layout()
        
        if write_folder:
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            plt.savefig(os.path.join(savedir, "zvariancevsz.pdf"))
        
        
        low_dim_latent=density.extract_latent(_x,earliest_latent=True)
        low_dim_latent=low_dim_latent.to(self._device)      
        ld=low_dim_latent.shape[1]
        self.set_seed(s=1454)
        # low_dim_latent=torch.randn(bs,ld)
        # low_dim_latent=torch.linspace(-3,3,bs).unsqueeze(-1).repeat(1,ld)

        # low_dim_latent=low_dim_latent.to(self._device)
        # This part probably does not matter becayse we do everything anyway
        reverse_sort_index = sorted(range(ld), key=var_sort_index.__getitem__)
           
        latent_dims_index=np.arange(ld)
        variances_zfromx=np.zeros_like(variances_z)
        # print(latent_dims_index, num_of_dims)
        num_of_dims=ld
        prominent_subgroups_index = np.array_split(latent_dims_index, num_of_dims)
        # print(prominent_subgroups_index)
        # make it random?
        low_dim_latent= torch.randn(low_dim_latent.size())
        low_dim_latent=low_dim_latent.to(self._device)      

        for i, subgroup in enumerate(prominent_subgroups_index):
            excluded_subgroup = list(set(latent_dims_index) - set(subgroup))
            low_dim_latent_=low_dim_latent[:,var_sort_index]#largest is last
            low_dim_latent_[:,excluded_subgroup]=torch.zeros(len(excluded_subgroup)).to(self._device)  
            low_dim_latent_=low_dim_latent_[:,reverse_sort_index]#put them back to their location
            x = density.fixed_sample(low_dim_latent_)
            x_flattened = x.view(_x.shape[0], -1)
            
            # Calculate the mean for each feature
            # Normalize the variance by the mean squared (Coefficient of Variation squared)
            def normalize_and_mean(x_flattened):
                variances_zfromx_ = torch.var(x_flattened, dim=0, unbiased=False)
                means_zfromx = torch.mean(x_flattened, dim=0)
                variances=variances_zfromx_ / (means_zfromx ** 2)
                min_variance = torch.min(variances)
                max_variance = torch.max(variances)
                normalized_variances = (variances - min_variance) / (max_variance - min_variance)
                meaned=torch.mean(normalized_variances)
                if all(0 <= x <= 1 for x in normalized_variances):
                    print("All values are between 0 and 1.")
                else:
                    print("Not all values are between 0 and 1.")
                return meaned
            
            # variances_zfromx[i] = normalize_and_mean(x_flattened)
            variances_zfromx[i]=torch.mean(torch.var(x_flattened, dim=0, unbiased=False))
            # variances_zfromx[i]=torch.mean(torch.var(x_flattened, dim=1, unbiased=False))
            
        
        var_fromx_sort_index = np.argsort(np.abs(variances_zfromx))

        sorted_fromx_variances_z_desc = variances_zfromx[var_fromx_sort_index[::-1]]
        # reversed_variances = sorted_variances_z_desc[::-1]
        cumulative_variances = np.cumsum(sorted_variances_z_desc)
        normalized_cumulative_variances = cumulative_variances / cumulative_variances[-1]

        plt.plot(dimensions,normalized_cumulative_variances, 'k.')
        
        
        dimensions=np.arange(len(sorted_fromx_variances_z_desc))
    


        zvars_dict = dict(zip(dimensions,normalized_cumulative_variances))
        zvars_dict = {int(k): v.item() for k, v in zvars_dict.items()}


        writepath = os.path.join(savedir, "zvars_fromx.json")
        json_dump = json.dumps(zvars_dict, indent=4)
        
        with open(writepath, "w") as f:
            f.write(json_dump)
        fig, ax = plt.subplots(1, 1)

        plt.plot(dimensions,cumulative_variances, 'k.')
        ax.set_xlabel(r'latent dimension', fontsize=15)
        ax.set_ylabel(r'variance', fontsize=15)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc=1, fontsize=7)
        plt.title
        plt.tight_layout()
        
        if write_folder:
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            plt.savefig(os.path.join(savedir, "zvariancevs_fromx_bs_var_randomn.pdf"))
        


        return var_fromx_sort_index

    
    def plot_two_dim_manifold_for_largest_effective_d(self, g_kk_sort_index, density, epoch, write_folder=None, fixed_noise=None, extent=None, labels=None):
        
        MIN = -3
        MAX = 3
        N_GRID = 8

        extent=[MIN, MAX, MIN, MAX]
        labels=["$z_1$", "$z_2$"]
        
        x = np.linspace(MIN, MAX, N_GRID)
        y = np.linspace(MAX, MIN, N_GRID)

        xv, yv = np.meshgrid(x, y)
        xy = np.stack((xv.reshape(N_GRID*N_GRID, ), yv.reshape(N_GRID*N_GRID, )), axis=1)

        noise = torch.from_numpy(xy)  # (torch.device("cuda"), dtype=torch.float32)
        ld=len(g_kk_sort_index)
        numbers=torch.zeros(N_GRID**2,ld)
        
        # NEED TO FIX THIS, SEE OTHER METHOD FOR RIGHT USE OF INDEXING
        g_kk_sort_index=list(g_kk_sort_index.detach().cpu().numpy())
        index_1 = g_kk_sort_index.index(ld-1)
        index_2 = g_kk_sort_index.index(ld-2)
        numbers[:,index_1]=noise[:,0]
        numbers[:,index_2]=noise[:,1]
        
        numbers=numbers.to(self._device)
        
        imgs = density.fixed_sample(numbers)
        num_images = imgs.shape[0]
        num_rows = int(np.ceil(num_images / min(np.sqrt(num_images), 10)))

        # NOTE: This may not normalize exactly as we would like, since we might not
        #       observe the full range of pixel values. It may be best to specify the
        #       range as the standard {0, ..., 255}, but just leaving it for now.
        grid = torchvision.utils.make_grid(
            imgs, nrow=num_rows, pad_value=1,
            normalize=True, scale_each=True
        )
        grid_permuted = grid.permute((1,2,0))
        plt.imshow(grid_permuted.detach().cpu().numpy(), extent=extent)
        if labels:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])

        if write_folder:
            savedir = os.path.join(write_folder, "test_metric")
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            plt.savefig(os.path.join(savedir, "sample_2d_manifold_with_bestgkk.pdf"))
        else:
            self._writer.write_image("sample_2d_manifold_with_bestgkk", grid, global_step=epoch)  


    @torch.no_grad()
    def plot_fid_for_effective_z(self,sort_index, density, epoch, write_folder=None, fixed_noise=None, extent=None, labels=None, num_of_dims=5, bs=16):
        try:
            import tensorflow as tf
        except: 
            print("No module tensorflow, exiting <<plot_fid_for_effective_z>> ")
            return None
        
        density.eval()
        x=self._x[-bs:] 

        # determine ld and reverse index:
        low_dim_latent=density.extract_latent(x,earliest_latent=True)
        low_dim_latent=low_dim_latent.to(self._device)
        ld=low_dim_latent.shape[1]
        reverse_sort_index = sorted(range(ld), key=sort_index.__getitem__)

        # ALTERNATIVE:
        # noise=torch.randn(1,ld).to(self._device)
        # low_dim_latent=noise
        
        # CAN USE THIS HERE MAYBE NO MEMORY ISSUE AND CAN HAVE BETTER RESOLUTION HERE
        # latent_dims_index=np.arange(ld)
        # # print(latent_dims_index, num_of_dims)
        # prominent_subgroups_index = np.array_split(latent_dims_index, num_of_dims)
        # # print(prominent_subgroups_index)
    
        # for i, subgroup in enumerate(prominent_subgroups_index):
        #     excluded_subgroup = list(set(latent_dims_index) - set(subgroup))
        #     # excluded_subgroup = latent_dims_index != latent_dims_index
        #     print(i,excluded_subgroup)
        #     low_dim_latent_=low_dim_latent[:,sort_index]#largest is last
        #     low_dim_latent_[:,excluded_subgroup]=torch.zeros(len(excluded_subgroup)).to(self._device)  
        #     low_dim_latent_=low_dim_latent_[:,reverse_sort_index]#put them back to their location
        
        mult=int(ld/num_of_dims)
        dimensions=list(range(num_of_dims+1))
        dimensions=[item * mult for item in dimensions]
        
        print("## calulating FID and recon for effective dimensions {}".format(dimensions))
        
        init_fid=True
        recons=[]
        fids=[]
        for dims in dimensions:
            density.eval()
            if dims==0: dims=1
            test_i = 0
            # for (x_batch, labels) in tqdm(self._test_loader, desc="Getting activations") : # iterate batches
            for (x_batch, labels) in self._test_loader: # iterate batches
            # for batch, _ in tqdm(dataloader, desc="Getting activations", )
                bs = x_batch.shape[0]
                print("## Effective Dimension {} / {} Val. batch {} / {} ".format( dims , ld , test_i*bs, len(self._test_loader.dataset) ))

                low_dim_latent=density.extract_latent(x_batch,earliest_latent=True)
                low_dim_latent=low_dim_latent.to(self._device)
                low_dim_latent_=low_dim_latent[:,sort_index]#largest is last
                low_dim_latent_[:,:-dims]=torch.zeros(ld-dims) #set to zero
                low_dim_latent_=low_dim_latent_[:,reverse_sort_index] #put back
                
                xhat_batch = density.fixed_sample(low_dim_latent_)
                
                if test_i==0:
                    x_all=x_batch
                    xhat_all=xhat_batch
                else:
                    x_all = torch.cat((x_all, x_batch), 0)
                    xhat_all = torch.cat((xhat_all, xhat_batch), 0)

                x_all = torch.cat((x_all, x_batch), 0)
                xhat_all = torch.cat((xhat_all, xhat_batch), 0)

                del  low_dim_latent_, low_dim_latent
                test_i += 1
                if test_i == 2:break
            

            recon=torch.nn.functional.mse_loss(x_all,xhat_all)
            recons.append(recon.cpu().detach().numpy())
            # print(xhat_all.shape,x_all.shape)
            try:
                xhat_fid=torch.reshape(xhat_all,((xhat_all.shape[0],x.shape[2],x.shape[3],1)) ).cpu().detach().numpy()
                x_fid=torch.reshape(x_all,((x_all.shape[0],x.shape[2],x.shape[3],1)) ).cpu().detach().numpy()
                # fid_score = fid_from_samples((xhat_fid*0.5+0.5)*255, (x_fid*0.5+0.5)*255, init_fid)
                fid_score = fid_from_samples(xhat_fid, x_fid, init_fid)
                init_fid = False
            except ValueError:
                print("FID CALCULATION FAILED")
                fid_score = 0
            fids.append(fid_score)
            del  xhat_all, x_all
           
            
        savedir = os.path.join(write_folder, "test_metric/")
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        recon_dict = dict(zip(dimensions, recons))
        recon_dict = {k: v.item() for k, v in recon_dict.items()}

        print('MSEs:', recon_dict)
        savedir = os.path.join(write_folder, "test_metric/")
        writepath = os.path.join(savedir, "recon.json")
        json_dump = json.dumps(recon_dict, indent=4)
        
        with open(writepath, "w") as f:
            f.write(json_dump)

        fid_dict = dict(zip(dimensions, fids))
        fid_dict = {k: v.item() for k, v in fid_dict.items()}

        print("FIDs:", fid_dict)
        savedir = os.path.join(write_folder, "test_metric/")
        writepath = os.path.join(savedir, "fid.json")
        json_dump = json.dumps(fid_dict, indent=4)
        
        with open(writepath, "w") as f:
            f.write(json_dump)

        fig, ax = plt.subplots(1, 1)

        plt.plot(dimensions,recons, 'k.')
        ax.set_xlabel(r'latent dimension', fontsize=15)
        ax.set_ylabel(r'recon loss', fontsize=15)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc=1, fontsize=7)
        plt.title
        plt.tight_layout()
        
        if write_folder:
            plt.savefig(os.path.join(savedir, "lossvsld.pdf"))
            
        fig, ax = plt.subplots(1, 1)
            
        plt.plot(dimensions,fids, 'k.')
        ax.set_xlabel(r'latent dimension', fontsize=15)
        ax.set_ylabel(r'FID score', fontsize=15)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc=1, fontsize=7)
        plt.title
        plt.tight_layout()
        
        if write_folder:
            plt.savefig(os.path.join(savedir, "fidvsld.pdf"))
      
            

        
    @torch.no_grad()
    def plot_samples_prominent_z_indivitual(self,sort_index, density, epoch, write_folder=None, fixed_noise=None,  extent=None, labels=None,bs=30, num_of_dims=5, random_seed=int(5545)):
        density.eval()
        x=self._x[-bs:] 
        xdim=x.shape[1]
        ydim=x.shape[2]
        low_dim_latent=density.extract_latent(x,earliest_latent=True)
        low_dim_latent=low_dim_latent.to(self._device)      
        ld=low_dim_latent.shape[1]
        self.set_seed(s=1454)
        low_dim_latent=torch.randn(bs,ld)
        # low_dim_latent=torch.linspace(-3,3,bs).unsqueeze(-1).repeat(1,ld)

        low_dim_latent=low_dim_latent.to(self._device)
        # TODO why is this memory crushing with more x etc?
        reverse_sort_index = sorted(range(ld), key=sort_index.__getitem__)
           
        latent_dims_index=np.arange(ld)
        # print(latent_dims_index, num_of_dims)
        prominent_subgroups_index = np.array_split(latent_dims_index, num_of_dims)
        # print(prominent_subgroups_index)
    
        for i, subgroup in enumerate(prominent_subgroups_index):
            excluded_subgroup = list(set(latent_dims_index) - set(subgroup))
            # excluded_subgroup = latent_dims_index != latent_dims_index
            print(i,subgroup)
            low_dim_latent_=low_dim_latent[:,sort_index]#largest is last
            low_dim_latent_[:,excluded_subgroup]=torch.zeros(len(excluded_subgroup)).to(self._device)  
            low_dim_latent_=low_dim_latent_[:,reverse_sort_index]#put them back to their location
            imgs = density.fixed_sample(low_dim_latent_)
            
            
            
     
            new_images=imgs
            if i==0:
                images=new_images
            else:
                images=torch.concat((images,new_images),axis=0)

            

        imgs=images
        num_rows = bs #+1
    
        grid = torchvision.utils.make_grid(
            imgs, nrow=num_rows, pad_value=1,
            normalize=True, scale_each=True
        )
    
        grid_permuted = grid.permute((1,2,0))
        axes=plt.imshow(grid_permuted.detach().cpu().numpy(), extent=None)

        # Disable shared axes aspect ratio adjustment
        axes.axes.set_aspect('auto')
                             
        # plt.axis('off')
        if write_folder:
            savedir = os.path.join(write_folder, "plotted_samples_prominent_d")
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            plt.savefig(os.path.join(savedir, "samples_indivitual.pdf"))  # Adjust the dpi value as needed
            # plt.savefig(os.path.join(savedir, "samples_indivitual.pdf"), dpi=1200, bbox_inches='tight', pad_inches=0, format='pdf')
        else:
            self._writer.write_image("samples_for_effective_ld", grid, global_step=epoch)  
 
    @torch.no_grad()
    def plot_samples_prominent_z_cumulative(self,sort_index, density, epoch, write_folder=None, fixed_noise=None,  extent=None, labels=None,bs=10, num_of_dims=5, random_seed=int(5545)):
        density.eval()
        x=self._x[-bs:] 
        xdim=x.shape[1]
        ydim=x.shape[2]
        low_dim_latent=density.extract_latent(x,earliest_latent=True)
        low_dim_latent=low_dim_latent.to(self._device)      
        ld=low_dim_latent.shape[1]
        self.set_seed(s=14545)
        low_dim_latent=torch.randn(bs,ld)

        low_dim_latent=low_dim_latent.to(self._device)
        # TODO why is this memory crushing with more x etc?
        reverse_sort_index = sorted(range(ld), key=sort_index.__getitem__)
        latent_dims_index=np.asarray(list(range(ld-1, -1, -1)))
        # latent_dims_index=np.arange(ld)
        prominent_subgroups_index = np.array_split(latent_dims_index, num_of_dims)
    
        # print(latent_dims_index, num_of_dims)
        # print(prominent_subgroups_index)
    
        for i, subgroup in enumerate(prominent_subgroups_index):
            if i==0:subgroup_=subgroup
            else:subgroup_=np.concatenate((subgroup_,subgroup))
            excluded_subgroup = list(set(latent_dims_index) - set(subgroup_))
            # excluded_subgroup = latent_dims_index != latent_dims_index
            print(i,subgroup_)
            low_dim_latent_=low_dim_latent[:,sort_index]#largest is last
            low_dim_latent_[:,excluded_subgroup]=torch.zeros(len(excluded_subgroup)).to(self._device)  
            low_dim_latent_=low_dim_latent_[:,reverse_sort_index]#put them back to their location
            imgs = density.fixed_sample(low_dim_latent_)
            
            
            new_images=imgs
            if i==0:
                images=new_images
            else:
                images=torch.concat((images,new_images),axis=0)
    
        imgs=images
        num_rows = bs
    
        grid = torchvision.utils.make_grid(
            imgs, nrow=num_rows, pad_value=1,
            normalize=True, scale_each=True
        )
    
        grid_permuted = grid.permute((1,2,0))
        axes=plt.imshow(grid_permuted.detach().cpu().numpy(), extent=extent)
        axes.axes.set_aspect('auto')

        plt.axis('off')
        if write_folder:
            savedir = os.path.join(write_folder, "plotted_samples_prominent_d")
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            plt.savefig(os.path.join(savedir, "samples_cumulative.pdf"), dpi=1200)  # Adjust the dpi value as needed
        else:
            self._writer.write_image("samples_for_effective_ld", grid, global_step=epoch)  
                           
            
    @torch.no_grad()
    def plot_samples_prominent_z_combined(self,sort_index, density, epoch, write_folder=None, fixed_noise=None,  extent=None, labels=None,bs=10, num_of_dims=5, random_seed=int(5545)):
        density.eval()
        x=self._x[-bs:] 
        xdim=x.shape[1]
        ydim=x.shape[2]
        low_dim_latent=density.extract_latent(x,earliest_latent=True)
        low_dim_latent=low_dim_latent.to(self._device)      
        ld=low_dim_latent.shape[1]
        self.set_seed()
        low_dim_latent=torch.randn(bs,ld)
        low_dim_latent=low_dim_latent.to(self._device)
        # TODO why is this memory crushing with more x etc?
        reverse_sort_index = sorted(range(ld), key=sort_index.__getitem__)
           
        latent_dims_index=np.arange(ld)
        # print(latent_dims_index, num_of_dims)
        prominent_subgroups_index = np.array_split(latent_dims_index, num_of_dims)
        # print(prominent_subgroups_index)
        
        for start, subgroup in enumerate(prominent_subgroups_index):
            for end in range(start, len(prominent_subgroups_index)):
                    subgroup_subset = np.concatenate(prominent_subgroups_index[start:end + 1])
                    excluded_subgroup = list(set(latent_dims_index) - set(subgroup_subset))
                    # excluded_subgroup = latent_dims_index != latent_dims_index
                    print(start,end,subgroup_subset)
                    low_dim_latent_=low_dim_latent[:,sort_index]#largest is last
                    low_dim_latent_[:,excluded_subgroup]=torch.zeros(len(excluded_subgroup)).to(self._device)  
                    low_dim_latent_=low_dim_latent_[:,reverse_sort_index]#put them back to their location
                    imgs = density.fixed_sample(low_dim_latent_)
                    num_squares=ld
                    rows = int(np.ceil(np.sqrt(num_squares)))
                    cols = int(np.ceil(num_squares / rows))
                    
                    # Create an index array of size num_squares
                    index_array = np.arange(num_squares)
                            
                    square_size = imgs.shape[2] // max(rows, cols)
                    scaling_factor = 0.99  # Adjust the scaling factor as desired (e.g., 0.8 makes the squares 80% of the original size)
                    new_square_size = int(square_size * scaling_factor)
                    
                    # Calculate the size of the black lines
                    line_thickness = square_size - new_square_size            # Create a 28x28 image with white squares
                    indicator_image = torch.ones((imgs.shape[2], imgs.shape[2])).to(self._device) * 255 
                            
                    # Iterate over the index array to set dark squares
                    for index in index_array:
                        ii = index // cols
                        jj = index % cols
                        if index in subgroup_subset:
                            # Dark square
                            x = jj * square_size + (jj * line_thickness)
                            y = ii * square_size + (ii * line_thickness)
                            # indicator_image[y:y+square_size, x:x+square_size] = 0
                            x_end = x + new_square_size
                            y_end = y + new_square_size
                
                            # Set white squares
                            indicator_image[y:y_end, x:x_end] = 0
                                        

                                
                                
                    indicator_image = indicator_image.unsqueeze(0).unsqueeze(0)
                    new_images = torch.cat((imgs, indicator_image), dim=0)
                    # new_images = torch.concat((imgs, indicator_image), axis=0)           
                    
                    if start==0:
                        images=new_images
                    else:
                        images=torch.concat((images,new_images),axis=0)
                          

        imgs=images
        num_rows = bs + 1
    
        grid = torchvision.utils.make_grid(
            imgs, nrow=num_rows, pad_value=1,
            normalize=True, scale_each=True
        )
    
        grid_permuted = grid.permute((1,2,0))
        axes=plt.imshow(grid_permuted.detach().cpu().numpy(), extent=extent)
        axes.axes.set_aspect('auto')

        plt.axis('off')
        if write_folder:
            savedir = os.path.join(write_folder, "plotted_samples_prominent_d")
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            plt.savefig(os.path.join(savedir, "samples_sequential.pdf"), dpi=1200)  # Adjust the dpi value as needed
        else:
            self._writer.write_image("samples_for_effective_ld", grid, global_step=epoch)  
         
    @torch.no_grad()
    def plot_samples_prominent_z_hierarchical(self,sort_index, density, epoch, write_folder=None, fixed_noise=None,  extent=None, labels=None,bs=4, num_of_dims=5, random_seed=int(5545)):
        density.eval()
        x=self._x[-bs:] 
        xdim=x.shape[1]
        ydim=x.shape[2]
        low_dim_latent=density.extract_latent(x,earliest_latent=True)
        low_dim_latent=low_dim_latent.to(self._device)      
        ld=low_dim_latent.shape[1]
        self.set_seed(s=13232)
        # low_dim_latent=torch.randn(bs,ld)
        low_dim_latent=torch.linspace(-3,3,bs).unsqueeze(-1).repeat(1,ld)
        
        mult=int(ld/num_of_dims)
        or_low_dim_latent=torch.randn(bs,mult)
        or_low_dim_latent=or_low_dim_latent.to(self._device)

        # TODO why is this memory crushing with more x etc?
        reverse_sort_index = sorted(range(ld), key=sort_index.__getitem__)
           
        latent_dims_index=np.asarray(list(range(ld-1, -1, -1)))
        prominent_subgroups_index = np.array_split(latent_dims_index, num_of_dims)
            
        subgroup1=list(set(prominent_subgroups_index[0]))
        subgroup2=list(set(prominent_subgroups_index[1]))
        subgroup3=list(set(prominent_subgroups_index[2]))
        subgroup4=list(set(prominent_subgroups_index[4]))
        print(subgroup1,subgroup2, subgroup3, subgroup4)
        
        low_dim_latent_=torch.zeros(bs,ld) # tree layers so 4*2*
        low_dim_latent_=low_dim_latent_.to(self._device) # tree layers so 4*2*
        low_dim_latent_[:,subgroup1]=or_low_dim_latent[:]
        _low_dim_latent_=low_dim_latent_[:,reverse_sort_index]#put them back to their location assuming higest is last
        img1 = density.fixed_sample(_low_dim_latent_)
        
        dummy_img=torch.ones(14,1,28,28).to(self._device) *255
        img1=torch.cat([dummy_img, img1, dummy_img], dim=0)
        

        low_dim_latent_=torch.zeros(bs*2,ld).to(self._device) # tree layers so 4*2*2
        low_dim_latent_=low_dim_latent_[:,sort_index]

        # combined_array = np.concatenate((prominent_subgroups_index[0], prominent_subgroups_index[1]))
        for j in range(bs*2):
            low_dim_latent_[j,subgroup1]=or_low_dim_latent[j//2]
            low_dim_latent_[j,subgroup2]=or_low_dim_latent[j%2]
        
        _low_dim_latent_=low_dim_latent_[:,reverse_sort_index]#put them back to their location assuming higest is last
        img2 = density.fixed_sample(_low_dim_latent_)
        
        dummy_img=torch.ones(12,1,28,28).to(self._device)*255
        img2=torch.cat([dummy_img, img2, dummy_img], dim=0)
        
        low_dim_latent_=torch.zeros(bs*2*2,ld).to(self._device) # tree layers so 4*2*2
        low_dim_latent_=low_dim_latent_[:,sort_index] #fake sort such as it works later
        for j in range(bs*2*2):
            low_dim_latent_[j,subgroup1]=or_low_dim_latent[j//4]
            low_dim_latent_[j,subgroup2]=or_low_dim_latent[j//2%2]
            low_dim_latent_[j,subgroup3]=or_low_dim_latent[j%2]

        _low_dim_latent_=low_dim_latent_[:,reverse_sort_index]#put them back to their location assuming higest is last
        img3 = density.fixed_sample(_low_dim_latent_)
      
        dummy_img=torch.ones(8,1,28,28).to(self._device)*255
        img3=torch.cat([dummy_img, img3, dummy_img], dim=0)
        
        low_dim_latent_=torch.zeros(bs*2*2*2,ld).to(self._device) # tree layers so 4*2*2
        low_dim_latent_=low_dim_latent_[:,sort_index] #fake sort such as it works later
        for j in range(bs*2*2*2):
            low_dim_latent_[j,subgroup1]=or_low_dim_latent[j//8]
            low_dim_latent_[j,subgroup2]=or_low_dim_latent[j//4%2]
            low_dim_latent_[j,subgroup3]=or_low_dim_latent[j//2%2]
            low_dim_latent_[j,subgroup4]=or_low_dim_latent[j%2]

        _low_dim_latent_=low_dim_latent_[:,reverse_sort_index]#put them back to their location assuming higest is last
        img4 = density.fixed_sample(_low_dim_latent_)

        images=torch.cat([img1, img2, img3, img4], dim=0)

        imgs=images
        num_rows = bs*2*2*2
    
        grid = torchvision.utils.make_grid(
            imgs, nrow=num_rows, pad_value=1,
            normalize=True, scale_each=True
        )
    
        grid_permuted = grid.permute((1,2,0))
        axes=plt.imshow(grid_permuted.detach().cpu().numpy(), extent=extent)
        axes.axes.set_aspect('auto')

        plt.axis('off')
        if write_folder:
            savedir = os.path.join(write_folder, "plotted_samples_prominent_d")
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            plt.savefig(os.path.join(savedir, "samples_hierarchical.pdf"), dpi=1200)  # Adjust the dpi value as needed
        else:
            self._writer.write_image("samples_for_effective_ld", grid_permuted, global_step=epoch)  
 
    @torch.no_grad()
    def plot_samples_prominent_z_combined_manifold_plotting(self, sort_index, density, epoch, write_folder=None, fixed_noise=None, extent=None, labels=None, bs=10, num_of_dims=10, random_seed=5545, zrange=3):
            density.eval()
            x = self._x[-bs:]
            ydim = x.shape[2]  # Not used in provided code snippet
            low_dim_latent = density.extract_latent(x, earliest_latent=True).to(self._device)
            
            ld = low_dim_latent.shape[1]
            num_of_dims=int(ld)
            self.set_seed(random_seed)  # Assuming this method is defined to set the random seed
    
            # Define low_dim_latent using batch_space for simplified dimension handling
            batch_space = torch.linspace(-zrange, zrange, steps=bs).unsqueeze(1).repeat(1, ld).to(self._device)
            
            reverse_sort_index = sorted(range(ld), key=sort_index.__getitem__)
            latent_dims_index = np.arange(ld)
            prominent_subgroups_index = np.array_split(latent_dims_index, num_of_dims)
            
            all_images = []
    
            for start in range(num_of_dims):
                # subgroup_subset = np.concatenate(prominent_subgroups_index[:start+1])
                subgroup_subset = prominent_subgroups_index[start]
                excluded_subgroup = list(set(latent_dims_index) - set(subgroup_subset))
                print(subgroup_subset, excluded_subgroup)
                modified_latent = batch_space.clone()
                modified_latent = modified_latent[:,sort_index]#largest is last
                modified_latent[:, excluded_subgroup] = 0
                imgs = density.fixed_sample(modified_latent[:, reverse_sort_index])
                
                # Calculate indicator image
                num_squares = ld
                square_size = imgs.shape[2] // int(np.sqrt(num_squares))
                scaling_factor = 0.99
                new_square_size = int(square_size * scaling_factor)
                line_thickness = square_size - new_square_size
                indicator_image = torch.ones((imgs.shape[2], imgs.shape[2])).to(self._device) * 255
    
                for index in range(num_squares):
                    ii, jj = divmod(index, int(np.sqrt(num_squares)))
                    x_start = jj * square_size + jj * line_thickness
                    y_start = ii * square_size + ii * line_thickness
                    indicator_image[y_start:y_start+new_square_size, x_start:x_start+new_square_size] = 0 if index in subgroup_subset else 255
    
                # Concatenate original and indicator images for the batch
                indicator_image = indicator_image.unsqueeze(0).unsqueeze(0)
                new_images = torch.cat([imgs, indicator_image], dim=0)
                all_images.append(new_images)
    
            # Combine all manipulated images into one tensor
            images = torch.cat(all_images, dim=0)
            grid = torchvision.utils.make_grid(images, nrow=bs + 1, pad_value=1, normalize=True, scale_each=True).permute(1, 2, 0)
    
            # Plotting
            plt.imshow(grid.cpu().numpy(), extent=extent)
            plt.axis('off')
    
            if write_folder:
                savedir = os.path.join(write_folder, "plotted_samples_prominent_d")
                os.makedirs(savedir, exist_ok=True)
                plt.savefig(os.path.join(savedir, f"samples_zrange_{zrange}.pdf"), dpi=1200)
            else:
                # Assuming _writer is an initialized and available object within the class
                self._writer.write_image("samples_for_effective_ld", grid, global_step=epoch)

    @torch.no_grad()
    def plot_samples_prominent_z_hierarchical(self,sort_index, density, epoch, write_folder=None, fixed_noise=None,  extent=None, labels=None,bs=4, num_of_dims=5, random_seed=int(5545)):
        density.eval()
        x=self._x[-bs:] 
        xdim=x.shape[1]
        ydim=x.shape[2]
        low_dim_latent=density.extract_latent(x,earliest_latent=True)
        low_dim_latent=low_dim_latent.to(self._device)      
        ld=low_dim_latent.shape[1]
        self.set_seed(s=13232)
        # low_dim_latent=torch.randn(bs,ld)
        low_dim_latent=torch.linspace(-3,3,bs).unsqueeze(-1).repeat(1,ld)
        
        mult=int(ld/num_of_dims)
        or_low_dim_latent=torch.randn(bs,mult)
        or_low_dim_latent=or_low_dim_latent.to(self._device)

        # TODO why is this memory crushing with more x etc?
        reverse_sort_index = sorted(range(ld), key=sort_index.__getitem__)
           
        latent_dims_index=np.asarray(list(range(ld-1, -1, -1)))
        # latent_dims_index=np.arange(ld)
        prominent_subgroups_index = np.array_split(latent_dims_index, num_of_dims)
    
            
        subgroup1=list(set(prominent_subgroups_index[0]))
        subgroup2=list(set(prominent_subgroups_index[1]))
        subgroup3=list(set(prominent_subgroups_index[2]))
        subgroup4=list(set(prominent_subgroups_index[4]))
        print(subgroup1,subgroup2, subgroup3, subgroup4)
        
        low_dim_latent_=torch.zeros(bs,ld) # tree layers so 4*2*
        low_dim_latent_=low_dim_latent_.to(self._device) # tree layers so 4*2*
        low_dim_latent_[:,subgroup1]=or_low_dim_latent[:]
        _low_dim_latent_=low_dim_latent_[:,reverse_sort_index]#put them back to their location assuming higest is last
        img1 = density.fixed_sample(_low_dim_latent_)
        
        dummy_img=torch.ones(14,1,28,28).to(self._device) *255
        img1=torch.cat([dummy_img, img1, dummy_img], dim=0)
        

        low_dim_latent_=torch.zeros(bs*2,ld).to(self._device) # tree layers so 4*2*2
        low_dim_latent_=low_dim_latent_[:,sort_index]

        # combined_array = np.concatenate((prominent_subgroups_index[0], prominent_subgroups_index[1]))
        for j in range(bs*2):
            low_dim_latent_[j,subgroup1]=or_low_dim_latent[j//2]
            low_dim_latent_[j,subgroup2]=or_low_dim_latent[j%2]
        
        _low_dim_latent_=low_dim_latent_[:,reverse_sort_index]#put them back to their location assuming higest is last
        img2 = density.fixed_sample(_low_dim_latent_)
        
        dummy_img=torch.ones(12,1,28,28).to(self._device)*255
        img2=torch.cat([dummy_img, img2, dummy_img], dim=0)
        
        low_dim_latent_=torch.zeros(bs*2*2,ld).to(self._device) # tree layers so 4*2*2
        low_dim_latent_=low_dim_latent_[:,sort_index] #fake sort such as it works later
        for j in range(bs*2*2):
            low_dim_latent_[j,subgroup1]=or_low_dim_latent[j//4]
            low_dim_latent_[j,subgroup2]=or_low_dim_latent[j//2%2]
            low_dim_latent_[j,subgroup3]=or_low_dim_latent[j%2]

        _low_dim_latent_=low_dim_latent_[:,reverse_sort_index]#put them back to their location assuming higest is last
        img3 = density.fixed_sample(_low_dim_latent_)
      
        dummy_img=torch.ones(8,1,28,28).to(self._device)*255
        img3=torch.cat([dummy_img, img3, dummy_img], dim=0)
        
        low_dim_latent_=torch.zeros(bs*2*2*2,ld).to(self._device) # tree layers so 4*2*2
        low_dim_latent_=low_dim_latent_[:,sort_index] #fake sort such as it works later
        for j in range(bs*2*2*2):
            low_dim_latent_[j,subgroup1]=or_low_dim_latent[j//8]
            low_dim_latent_[j,subgroup2]=or_low_dim_latent[j//4%2]
            low_dim_latent_[j,subgroup3]=or_low_dim_latent[j//2%2]
            low_dim_latent_[j,subgroup4]=or_low_dim_latent[j%2]

        _low_dim_latent_=low_dim_latent_[:,reverse_sort_index]#put them back to their location assuming higest is last
        img4 = density.fixed_sample(_low_dim_latent_)

        images=torch.cat([img1, img2, img3, img4], dim=0)

        imgs=images
        num_rows = bs*2*2*2
    
        grid = torchvision.utils.make_grid(
            imgs, nrow=num_rows, pad_value=1,
            normalize=True, scale_each=True
        )
    
        grid_permuted = grid.permute((1,2,0))
        axes=plt.imshow(grid_permuted.detach().cpu().numpy(), extent=extent)
        axes.axes.set_aspect('auto')

        plt.axis('off')
        if write_folder:
            savedir = os.path.join(write_folder, "plotted_samples_prominent_d")
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            plt.savefig(os.path.join(savedir, "samples_hierarchical.pdf"), dpi=1200)  # Adjust the dpi value as needed
        else:
            self._writer.write_image("samples_for_effective_ld", grid_permuted, global_step=epoch)  
 
            
            
    def plot_pairwise_dimension_comparison(self,sort_index, density, epoch, write_folder=None, fixed_noise=None, extent=None, labels=None):
        try:
            import seaborn
            import pandas
        except: 
            print("No module seaborn or pandas")
            return None
        
        density.eval()

        x=self._x[-128:] 
        low_dim_latent=density.extract_latent(x,earliest_latent=True)
        # print(x.shape)
        data = pandas.DataFrame(low_dim_latent.detach().cpu().numpy())
        plt.figure()
        seaborn.pairplot(data,corner=False)

        if write_folder:
            savedir = os.path.join(write_folder, "test_metric")
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            plt.savefig(os.path.join(savedir, "pairwise.pdf"))
            

        correlation_matrix = data.corr().abs()
        average_correlation = correlation_matrix.values[correlation_matrix.values != 1].mean()
        plt.figure(figsize=(10, 8))
        print("PAIRWISE:",low_dim_latent.shape)

        if low_dim_latent.shape[1] > 11: 
            annot_bool=False
        else:annot_bool=True
        seaborn.heatmap(correlation_matrix, annot=annot_bool, vmin=0, vmax=0.5, center=0.25, cmap='viridis')
        plt.text(
            x=0.5, y=-0.1, s=f"Average Correlation: {average_correlation:.5f}",
            fontsize=12, ha='center', va='bottom', transform=plt.gca().transAxes
)
        if write_folder:
            savedir = os.path.join(write_folder, "test_metric")
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            plt.savefig(os.path.join(savedir, "pairwise_corelation.pdf"), dpi=1200)            
        # else:
            # self._writer.write_image("pairwise", grid, global_step=epoch)  
        

    def plot_expectation_zero(self, sort_index, density, epoch, write_folder=None, fixed_noise=None, extent=None, labels=None):
        ld=len(sort_index)
        numbers=torch.zeros(64,ld)
        noise = numbers.to(self._device) #torch.device("cuda"), dtype=torch.float32)
        imgs = density.fixed_sample(noise)
        num_images = imgs.shape[0]
        num_rows = int(np.ceil(num_images / min(np.sqrt(num_images), 10)))

        grid = torchvision.utils.make_grid(
            imgs, nrow=num_rows, pad_value=1,
            normalize=True, scale_each=True
        )
        grid_permuted = grid.permute((1,2,0))
        plt.imshow(grid_permuted.detach().cpu().numpy(), extent=extent)
        if labels:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])

        if write_folder:
            savedir = os.path.join(write_folder, "test_metric")
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            plt.savefig(os.path.join(savedir, "samples_at_zero.pdf"))
        else:
            self._writer.write_image("samples_at_zero", grid, global_step=epoch)  
 
    def plot_individual_latent(self, sort_index, density, epoch, write_folder=None, fixed_noise=None, extent=None, labels=None):
        # numbers = torch.column_stack((torch.randn(20), torch.zeros(20,19)) )
        ld=len(sort_index)
        numbers= torch.eye(ld)*torch.randn(1)
        noise = numbers.to(self._device)
        imgs = density.fixed_sample(noise)
        num_images = imgs.shape[0]
        num_rows = int(np.ceil(num_images / min(np.sqrt(num_images), 10)))
        grid = torchvision.utils.make_grid(
            imgs, nrow=num_rows, pad_value=1,
            normalize=True, scale_each=True
        )
        grid_permuted = grid.permute((1,2,0))
        plt.imshow(grid_permuted.detach().cpu().numpy(), extent=extent)
        if labels:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])

        if write_folder:
            savedir = os.path.join(write_folder, "test_metric")
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            plt.savefig(os.path.join(savedir, "samples_at_individual_zi.pdf"))
        else:
            self._writer.write_image("samples_at_individual_zi", grid, global_step=epoch)              

    def jacobian(self,y: torch.Tensor, x: torch.Tensor, create_graph=False):
        jac=torch.ones([y.shape[0],y.shape[1],x.shape[1]])
        for i in range(y.shape[1]):
            # for j in range(x.shape[1]):
                batched_grad = torch.ones_like(y.select(1,i))
                grad, = torch.autograd.grad(y.select(1,i),x,grad_outputs=batched_grad, is_grads_batched=False,  retain_graph=True, create_graph=False, allow_unused=False) 
                jac[:,i,:]=grad
        return jac

class ImageCenteringDensityVisualizer(DensityVisualizer):
    def __init__(self, writer, x_train, device, num_elbo_samples, test_loader=None):
        super().__init__(writer=writer)

        self._x = x_train
        self._device = device
        self._num_elbo_samples = num_elbo_samples
        self._test_loader = test_loader
        
    def set_seed(self, s=42):
        np.random.seed(s)
        random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(s)
        print(f"Random seed set as {s}")
    
    
    # @torch.no_grad()
    def visualize(self, density, epoch, write_folder=None, fixed_noise=None, extent=None, labels=None):
        density.eval()
        self.compare_centers(density,write_folder)
    
    def compare_centers(self, density,write_folder):
        _x=self._x[-64:]
        _x.requires_grad=True
        low_dim_latent=density.extract_latent(_x,earliest_latent=True)
        print( "E(x)={},E(z)={}".format(_x.mean(),low_dim_latent.mean() ))

# metricplots-end----------------------------------------------------------------------------------- 
      
      
class TwoDimensionalVisualizerBase(DensityVisualizer):
    _NUM_TRAIN_POINTS_TO_SHOW = 500
    _BATCH_SIZE = 1000
    _CMAP = "plasma"
    _CMAP_LL = "autumn_r"
    _CMAP_D = "cool"
    _PADDING = .2
    _FS=15

    def __init__(self, writer, x_train, device):
        super().__init__(writer=writer)

        self._x = x_train
        self._device = device

    def _lims(self, t):
        return (
            t.min().item() - self._PADDING,
            t.max().item() + self._PADDING
        )

    def _plot_x_train(self):
        x = self._x.cpu()
        x = x[torch.randint(high=x.shape[0], size=(self._NUM_TRAIN_POINTS_TO_SHOW,))]
        plt.scatter(x[:, 0], x[:, 1], c="k", marker=".", s=7, linewidth=0.5, alpha=0.2)

    def _plot_density(self, density):
        raise NotImplementedError

    def visualize(self, density, epoch, write_folder=None):
        self._plot_density(density)
        # self._plot_x_train()

        if write_folder:
            plt.savefig(os.path.join(write_folder, "density.pdf"), bbox_inches='tight')
        else:
            self._writer.write_figure("density", plt.gcf(), epoch)

        plt.close()


class TwoDimensionalDensityVisualizer(TwoDimensionalVisualizerBase):
    _GRID_SIZE = 150
    _CONTOUR_LEVELS = 50

    def __init__(self, writer, x_train, num_elbo_samples, device):
        super().__init__(writer, x_train, device)

        self._num_elbo_samples = num_elbo_samples

        x1_lims = self._lims(self._x[:, 0])
        x2_lims = self._lims(self._x[:, 1])

        self._grid_x1, self._grid_x2 = torch.meshgrid((
            torch.linspace(*x1_lims, self._GRID_SIZE),
            torch.linspace(*x2_lims, self._GRID_SIZE)
        ))

        x1_x2 = torch.stack((self._grid_x1, self._grid_x2), dim=2).view(-1, 2)

        self._loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x1_x2.to(self._device)),
            batch_size=self._BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=False
        )

    def _plot_density(self, density):
        probs = []
        for x1_x2_batch, in tqdm.tqdm(self._loader, leave=False, desc="Plotting"):
            with torch.no_grad():
                log_prob = metrics(density, x1_x2_batch, self._num_elbo_samples)["log-prob"]
            probs.append(torch.exp(log_prob))

        probs = torch.cat(probs, dim=0).view(*self._grid_x1.shape).cpu()

        plt.figure()

        contours = plt.contourf(self._grid_x1, self._grid_x2, probs, levels=self._CONTOUR_LEVELS, cmap=self._CMAP)
        for c in contours.collections:
            c.set_edgecolor("face")
        cb = plt.colorbar()
        cb.solids.set_edgecolor("face")


class TwoDimensionalNonSquareVisualizer(TwoDimensionalVisualizerBase):
    _LINSPACE_SIZE = 1000
    _LINSPACE_LIMITS = [-3, 3]

    def __init__(self, writer, x_train, device, log_prob_low, log_prob_high):
        super().__init__(writer, x_train, device)
        self._log_prob_limits = [log_prob_low, log_prob_high]
        self._low_dim_space = torch.linspace(*self._LINSPACE_LIMITS, self._LINSPACE_SIZE)

    def visualize(self, density, epoch, write_folder=None):
        self._embedded_manifold = density.fixed_sample(self._low_dim_space.unsqueeze(1))

        super().visualize(density, epoch, write_folder)

        self._plot_manifold_distance(density)
        self._plot_x_train()

        if write_folder:
            plt.savefig(os.path.join(write_folder, "distances.pdf"), bbox_inches='tight')
        else:
            self._writer.write_figure("manifold-distances", plt.gcf(), epoch)

        plt.close()

        self._plot_pullback_density(density, write_folder)
        if write_folder:
            plt.savefig(os.path.join(write_folder, "pullback.pdf"), bbox_inches='tight')
        else:
            self._writer.write_figure("pullback-density", plt.gcf(), epoch)

        plt.close()

        self._plot_ground_truth(density, write_folder)
        self._plot_x_train()

        if write_folder:
            plt.savefig(os.path.join(write_folder, "ground_truth.pdf"), bbox_inches='tight')
        else:
            self._writer.write_figure("ground-truth", plt.gcf(), epoch)

        plt.close()

    def _plot_density(self, density):
        log_probs = density.elbo(
            x=self._embedded_manifold,
            add_reconstruction=False,
            likelihood_wt=1.,
            visualization=True
        )["elbo"].squeeze()

        self._plot_along_manifold(
            pytorch_colours=log_probs[1:],
            cbar_limits=self._log_prob_limits,
            metric="log-likelihood"
        )

    def _plot_manifold_distance(self, density):
        squared_distances = ((self._embedded_manifold[:-1] - self._embedded_manifold[1:])**2).sum(axis=1)
        distances = torch.sqrt(squared_distances).detach()

        self._plot_along_manifold(
            pytorch_colours=distances,
            cbar_limits=[0, torch.max(distances)],
            metric="speed"
        )

    def _plot_along_manifold(self, pytorch_colours, metric, cbar_limits):
        fig, ax = plt.subplots(1, 1)
        plt.axis("off")

        colours = pytorch_colours.detach().numpy()

        xy = self._embedded_manifold.detach().numpy()
        xy_collection = np.concatenate([xy[:-1,np.newaxis,:], xy[1:,np.newaxis,:]], axis=1)

        cbar_min, cbar_max = cbar_limits
        cmap = self._CMAP
        if metric == "log-likelihood":
            cmap = self._CMAP_LL
            label = r'$\log p(x)$'
        if metric == "speed":
            cmap = self._CMAP_D
            label = r'speed'
        lc = LineCollection(
            xy_collection,
            cmap=cmap,
            norm=plt.Normalize(cbar_min, cbar_max),
            linewidths=3
        )
        lc.set_array(np.clip(colours, cbar_min, cbar_max))
        ax.add_collection(lc)
        axcb = fig.colorbar(lc, extend="both")
        axcb.set_label(label, fontsize=self._FS)

    def _plot_pullback_density(self, density, write_folder):
        log_jac_jac_t = density.pullback_log_jac_jac_transpose(self._embedded_manifold)

        circle_projections = self._embedded_manifold / torch.sqrt(torch.sum(self._embedded_manifold**2, dim=1, keepdim=True))
        log_groundtruth_numerator = circle_projections[:,1]

        norm_const = 2*np.pi*i0(1)

        probs_unnorm = torch.exp(log_groundtruth_numerator - 1/2*log_jac_jac_t).detach().cpu().numpy()
        probs = probs_unnorm/norm_const

        pullback_np = np.stack([self._low_dim_space.detach().cpu().numpy(), probs], axis=0)
        if write_folder:
            np.save(os.path.join(write_folder, "pullback.npy"), pullback_np)

        plt.plot(self._low_dim_space.detach().cpu().numpy(), probs)

    def _plot_ground_truth(self, density, write_folder):
        log_probs = vonmises.logpdf(np.linspace(-np.pi, np.pi, num=1000, endpoint=False), 1., loc=np.pi/2)

        self._plot_along_circle(
            density=density,
            colours=log_probs,
            cbar_limits=self._log_prob_limits,
            write_folder=write_folder
        )

    def _plot_along_circle(self, density, colours, cbar_limits, write_folder):
        fig, ax = plt.subplots(1, 1)
        plt.axis("off")

        theta = np.linspace(-np.pi, np.pi, num=1000, endpoint=False)
        xy = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        xy_collection = np.concatenate([xy[:-1, np.newaxis, :], xy[1:, np.newaxis, :]], axis=1)

        cbar_min, cbar_max = cbar_limits
        cmap = self._CMAP_LL
        lc = LineCollection(
            xy_collection,
            cmap=cmap,
            norm=plt.Normalize(cbar_min, cbar_max),
            linewidths=3
        )
        lc.set_array(np.clip(colours, cbar_min, cbar_max))
        ax.add_collection(lc)
        axcb = fig.colorbar(lc, extend="both")
        axcb.set_label(r'$\log p(x)$', fontsize=self._FS)

        theta = vonmises.rvs(1, size=1000, loc=np.pi/2)
        xy2 = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        xy_torch = torch.tensor(xy2, dtype=torch.float32)
        z = density._extract_latent(xy_torch, **{"earliest_latent": True}).detach().cpu().numpy().reshape(-1)
        kde = gaussian_kde(z)
        xs = np.linspace(-3, 3, 1000)
        kde_np = np.stack([xs, kde.pdf(xs)], axis=0)
        if write_folder:
            np.save(os.path.join(write_folder, "kde.npy"), kde_np)
        # ax.plot(xs, kde.pdf(xs))



# KF:
class TwoDimensionalVisualizerBaseFullLatent(DensityVisualizer):
    _NUM_TRAIN_POINTS_TO_SHOW = 500
    _NUM_SAMPLE_POINTS_TO_SHOW = 500
    _NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW = 100

    _BATCH_SIZE = 1000
    # colormap='cool'
    colormap='plasma'
    _CMAP = colormap
    _CMAP_LL = "autumn_r"
    _CMAP_D = 'cool'
    _PADDING = .2
    _FS=15


    _XYZ_LIM_LEFT=-0.6
    _XYZ_LIM_RIGHT=0.6
    def __init__(self, writer, x_train, device):
        super().__init__(writer=writer)

        self._x = x_train
        self._device = device

    def _lims(self, t):
        return (
            t.min().item() - self._PADDING,
            t.max().item() + self._PADDING
        )

    def _plot_x_train(self):
        x = self._x.cpu()
        x = x[torch.randint(high=x.shape[0], size=(self._NUM_TRAIN_POINTS_TO_SHOW,))]
        plt.scatter(x[:, 0], x[:, 1], c="k", marker=".", s=7, linewidth=0.5, alpha=0.2)

    def _plot_density(self, density):
        raise NotImplementedError

    def visualize(self, density, epoch, write_folder=None):
        self._plot_density(density)
        # self._plot_x_train()

        if write_folder:
            plt.savefig(os.path.join(write_folder, "density.pdf"), bbox_inches='tight')
        else:
            self._writer.write_figure("density", plt.gcf(), epoch)

        plt.close()


class TwoDimensionalNonSquareVisualizer_2dlatent(TwoDimensionalVisualizerBaseFullLatent):
    _GRID_SIZE = 50
    _CONTOUR_LEVELS = 50

    def __init__(self, writer, x_train, num_elbo_samples, device):
        super().__init__(writer, x_train, device)

        self._num_elbo_samples = num_elbo_samples

        x1_lims = self._lims(self._x[:, 0])
        x2_lims = self._lims(self._x[:, 1])

        self._grid_x1, self._grid_x2 = torch.meshgrid((
            torch.linspace(*x1_lims, self._GRID_SIZE),
            torch.linspace(*x2_lims, self._GRID_SIZE)
        ))

        x1_x2 = torch.stack((self._grid_x1, self._grid_x2), dim=2).view(-1, 2)

        self._loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x1_x2.to(self._device)),
            batch_size=self._BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=False
        )

    def _plot_density(self, density):
        device = self._device

        x = self._x.cpu()
        x = x[torch.randint(high=x.shape[0], size=(self._NUM_TRAIN_POINTS_TO_SHOW,))]
        low_dim_space = torch.linspace(-2.5,2.5,self._NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW)
        numbers = [
        (torch.cat((torch.randn(self._NUM_SAMPLE_POINTS_TO_SHOW,1), torch.randn(self._NUM_SAMPLE_POINTS_TO_SHOW,1)), 1)),
        (torch.cat((low_dim_space[:,np.newaxis], torch.zeros(self._NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW,1)), 1)),
        (torch.cat((torch.zeros(self._NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW,1),low_dim_space[:,np.newaxis]), 1))
        ]
        numbers= [ t.to(device) for t in numbers ]

        labels=['(i)','(ii)','(iii)']
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig, axs = plt.subplots(1,3,figsize=plt.figaspect(0.33))
        fig.tight_layout()        
        for i in range(len(numbers)):
            axs[i].text(-2.0, 2.0, str(labels[i]), fontsize=self._FS,  verticalalignment='top')
            axs[i].grid(False)
            axs[i].axis(False)
            embedded_manifold = density.fixed_sample(numbers[i])
            log_probs = density.elbo(
                x=embedded_manifold,
                add_reconstruction=False,
                likelihood_wt=1.,
                visualization=True
            )["elbo"].squeeze()
            # print(log_probs.shape, embedded_manifold.shape)
            embedded_manifold=embedded_manifold.cpu().detach().numpy()
            # Scaling for plotting
            log_probs=log_probs.cpu().detach().numpy()
            scaler=preprocessing.MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(log_probs.reshape(-1, 1))
            log_probs=scaler.transform(log_probs.reshape(-1, 1)).ravel()

            # plt.title("sample both dims")
            axs[i].scatter(x[:, 0], x[:, 1], c="k", marker=".", s=5, linewidth=0.5, alpha=0.3)
            im=axs[i].scatter(embedded_manifold[:, 0], embedded_manifold[:, 1], c=log_probs, cmap=self._CMAP, marker="o", s=40)
            divider = make_axes_locatable(axs[i])

        cax = divider.append_axes('right', size='5%', pad=0.05)
        axcb=fig.colorbar(im, cax=cax, orientation='vertical',shrink=0.8)
        axcb.set_label(r'$\log p(x)$', fontsize=self._FS)
        fig.tight_layout()        
        fig.subplots_adjust(wspace=0, hspace=0)
        
        
        
class ThreeDimensionalVisualizerBase(DensityVisualizer):
        _NUM_TRAIN_POINTS_TO_SHOW = 500
        _NUM_SAMPLE_POINTS_TO_SHOW = 500
        _NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW = 100
    
        _BATCH_SIZE = 1000
        # colormap='cool'
        colormap='plasma'
        _CMAP = colormap
        _CMAP_LL = "autumn_r"
        _CMAP_D = 'cool'
        _PADDING = .2

        _XYZ_LIM_LEFT=-0.7
        _XYZ_LIM_RIGHT=0.7

        _FS=15

        def __init__(self, writer, x_train, device):
            super().__init__(writer=writer)
        
            self._x = x_train
            self._device = device
        
        def _lims(self, t):
            return (
                t.min().item() - self._PADDING,
                t.max().item() + self._PADDING
            )
        
        def _plot_x_train(self):
            x = self._x.cpu()
            x = x[torch.randint(high=x.shape[0], size=(self._NUM_TRAIN_POINTS_TO_SHOW,))]
            plt.scatter(x[:, 0], x[:, 1], x[:,2], c="k", marker=".", s=7, linewidth=0.5, alpha=0.2)
        
        def _plot_density(self, density):
            raise NotImplementedError   

        def visualize(self, density, epoch, write_folder=None):
            
            self._plot_density(density)
            if write_folder:
                plt.savefig(os.path.join(write_folder, "density.pdf"), bbox_inches='tight')
            else:
                self._writer.write_figure("density", plt.gcf(), epoch)
        
            plt.close()


class ThreeDimensionalNonSquareVisualizer_3dlatent(ThreeDimensionalVisualizerBase):
    _GRID_SIZE = 50
    _CONTOUR_LEVELS = 50

    def __init__(self, writer, x_train, num_elbo_samples, device):
        super().__init__(writer, x_train, device)

        self._num_elbo_samples = num_elbo_samples

        x1_lims = self._lims(self._x[:, 0])
        x2_lims = self._lims(self._x[:, 1])

        self._grid_x1, self._grid_x2 = torch.meshgrid((
            torch.linspace(*x1_lims, self._GRID_SIZE),
            torch.linspace(*x2_lims, self._GRID_SIZE)
        ))

        x1_x2 = torch.stack((self._grid_x1, self._grid_x2), dim=2).view(-1, 2)

        self._loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x1_x2.to(self._device)),
            batch_size=self._BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=False
        )
        
    def _plot_density(self, density):
        device =self._device

        x = self._x.cpu()
        x = x[torch.randint(high=x.shape[0], size=(self._NUM_TRAIN_POINTS_TO_SHOW,))]
        low_dim_space = torch.linspace(-2.5,2.5,self._NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW)
        numbers = [
        (torch.cat((torch.randn(self._NUM_SAMPLE_POINTS_TO_SHOW,1), torch.randn(self._NUM_SAMPLE_POINTS_TO_SHOW,1),torch.randn(self._NUM_SAMPLE_POINTS_TO_SHOW,1)), 1)),
        (torch.cat((low_dim_space[:,np.newaxis], torch.zeros(self._NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW,1), torch.zeros(self._NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW,1)),  1)),
        (torch.cat((torch.zeros(self._NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW,1),low_dim_space[:,np.newaxis], torch.zeros(self._NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW,1)), 1)),
        (torch.cat((torch.zeros(self._NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW,1), torch.zeros(self._NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW,1),low_dim_space[:,np.newaxis]),  1)),
        ]
        numbers= [ t.to(device) for t in numbers ]
        labels=['(i)','(ii)','(iii)', '(iv)']

        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig = plt.figure(figsize=plt.figaspect(0.33))
        for i in range(len(numbers)):
            ax = fig.add_subplot(1, 4, int(i+1), projection='3d')
            ax.grid(False)
            ax.axis(False)
            ax.text(-2,1,0, str(labels[i]), fontsize=self._FS )
            # ax.text()
            embedded_manifold = density.fixed_sample(numbers[i])
            log_probs = density.elbo(
                x=embedded_manifold,
                add_reconstruction=False,
                likelihood_wt=1.,
                visualization=True
            )["elbo"].squeeze()
            # print(log_probs.shape, embedded_manifold.shape)
            embedded_manifold=embedded_manifold.cpu().detach().numpy()
            log_probs=log_probs.cpu().detach().numpy()

            scaler=preprocessing.MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(log_probs.reshape(-1, 1))
            log_probs=scaler.transform(log_probs.reshape(-1, 1)).ravel()

            ax.scatter(x[:, 0], x[:, 1], x[:, 2], c="k", marker=".", s=7, linewidth=0.5, alpha=0.3)
            im=ax.scatter(embedded_manifold[:, 0], embedded_manifold[:, 1],embedded_manifold[:, 2], c=log_probs, cmap=self._CMAP, marker="o", s=40)

            
            if limitaxes:
                ax.axes.set_xlim3d(left=self._XYZ_LIM_LEFT, right=self._XYZ_LIM_RIGHT) 
                ax.axes.set_ylim3d(bottom=self._XYZ_LIM_LEFT, top=self._XYZ_LIM_RIGHT) 
                ax.axes.set_zlim3d(bottom=self._XYZ_LIM_LEFT, top=self._XYZ_LIM_RIGHT) 
#       [left, bottom, width, height]
        cbaxes = fig.add_axes([1, 0.15, 0.015, 0.7])
        axcb=fig.colorbar(im,cax=cbaxes, shrink=0.6)
        # cax = divider.append_axes('top', size='5%', pad=0.05)
        # axcb=fig.colorbar(im, cax=cax, orientation='vertical',shrink=0.8)
        axcb.set_label(r'$\log p(x)$', fontsize=self._FS)
        fig.tight_layout()        
        fig.subplots_adjust(wspace=0, hspace=0)
            
        
        
        

class ThreeDimensionalNonSquareVisualizer_2dlatent(ThreeDimensionalVisualizerBase):
    _GRID_SIZE = 50
    _CONTOUR_LEVELS = 50

    def __init__(self, writer, x_train, num_elbo_samples, device):
        super().__init__(writer, x_train, device)

        self._num_elbo_samples = num_elbo_samples

        x1_lims = self._lims(self._x[:, 0])
        x2_lims = self._lims(self._x[:, 1])

        self._grid_x1, self._grid_x2 = torch.meshgrid((
            torch.linspace(*x1_lims, self._GRID_SIZE),
            torch.linspace(*x2_lims, self._GRID_SIZE)
        ))

        x1_x2 = torch.stack((self._grid_x1, self._grid_x2), dim=2).view(-1, 2)

        self._loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x1_x2.to(self._device)),
            batch_size=self._BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=False
        )

        
    def _plot_density(self, density):

        x = self._x.cpu()
        x = x[torch.randint(high=x.shape[0], size=(self._NUM_TRAIN_POINTS_TO_SHOW,))]
        low_dim_space = torch.linspace(-2.5,2.5,self._NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW)
        numbers = [
        (torch.cat((torch.randn(self._NUM_SAMPLE_POINTS_TO_SHOW,1), torch.randn(self._NUM_SAMPLE_POINTS_TO_SHOW,1)), 1)),
        (torch.cat((low_dim_space[:,np.newaxis], torch.zeros(self._NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW,1)), 1)),
        (torch.cat((torch.zeros(self._NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW,1),low_dim_space[:,np.newaxis]), 1))
        ]        
        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # numbers=numbers.to(device, dtype=torch.float32)
        numbers= [ t.to(self._device) for t in numbers ]
        fig = plt.figure(figsize=plt.figaspect(3))
        for i in range(len(numbers)):
            ax = fig.add_subplot(3, 1, int(i+1), projection='3d')
            ax.grid(False)
            ax.axis(False)
            embedded_manifold = density.fixed_sample(numbers[i])
            log_probs = density.elbo(
                x=embedded_manifold,
                add_reconstruction=False,
                likelihood_wt=1.,
                visualization=True
            )["elbo"].squeeze()
            # print(log_probs.shape, embedded_manifold.shape)
            embedded_manifold=embedded_manifold.cpu().detach().numpy()
            log_probs=log_probs.cpu().detach().numpy()
            # plt.title("sample both dims")
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], c="k", marker=".", s=7, linewidth=0.5, alpha=0.2)
            im=ax.scatter(embedded_manifold[:, 0], embedded_manifold[:, 1],embedded_manifold[:, 2], c=log_probs, cmap=self._CMAP, marker="o", s=7)
            axcb= fig.colorbar(im,extend="both", shrink=0.8)
            axcb.set_label(r'$\log p(x)$', fontsize=self._FS)
            if limitaxes:
                ax.axes.set_xlim3d(left=self._XYZ_LIM_LEFT, right=self._XYZ_LIM_RIGHT) 
                ax.axes.set_ylim3d(bottom=self._XYZ_LIM_LEFT, top=self._XYZ_LIM_RIGHT) 
                ax.axes.set_zlim3d(bottom=self._XYZ_LIM_LEFT, top=self._XYZ_LIM_RIGHT) 

        fig.tight_layout()        
        fig.subplots_adjust(wspace=0, hspace=0)
        


        

class ThreeDimensionalNonSquareVisualizer_1dlatent(ThreeDimensionalVisualizerBase):
    _GRID_SIZE = 50
    _CONTOUR_LEVELS = 50

    def __init__(self, writer, x_train, num_elbo_samples, device):
        super().__init__(writer, x_train, device)

        self._num_elbo_samples = num_elbo_samples

        x1_lims = self._lims(self._x[:, 0])
        x2_lims = self._lims(self._x[:, 1])

        self._grid_x1, self._grid_x2 = torch.meshgrid((
            torch.linspace(*x1_lims, self._GRID_SIZE),
            torch.linspace(*x2_lims, self._GRID_SIZE)
        ))

        x1_x2 = torch.stack((self._grid_x1, self._grid_x2), dim=2).view(-1, 2)

        self._loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x1_x2.to(self._device)),
            batch_size=self._BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=False
        )
        
    def _plot_density(self, density):
        x = self._x.cpu()
        x = x[torch.randint(high=x.shape[0], size=(self._NUM_TRAIN_POINTS_TO_SHOW,))]
        low_dim_space = torch.linspace(-2.5,2.5,self._NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW)
        numbers = [
        torch.randn(self._NUM_SAMPLE_POINTS_TO_SHOW,1),
        low_dim_space[:,np.newaxis]
        ]
        
        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig = plt.figure(figsize=plt.figaspect(2))
        for i in range(len(numbers)):
            ax = fig.add_subplot(2, 1, int(i+1), projection='3d')
            ax.grid(False)
            ax.axis(False)
            embedded_manifold = density.fixed_sample(numbers[i])
            log_probs = density.elbo(
                x=embedded_manifold,
                add_reconstruction=False,
                likelihood_wt=1.,
                visualization=True
            )["elbo"].squeeze()
            # print(log_probs.shape, embedded_manifold.shape)
            embedded_manifold=embedded_manifold.cpu().detach().numpy()
            log_probs=log_probs.cpu().detach().numpy()
            # plt.title("sample both dims")
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], c="k", marker=".", s=7, linewidth=0.5, alpha=0.2)
            im=ax.scatter(embedded_manifold[:, 0], embedded_manifold[:, 1],embedded_manifold[:, 2], c=log_probs, cmap=self._CMAP, marker="o", s=7)
            axcb= fig.colorbar(im,extend="both", shrink=0.8)
            axcb.set_label(r'$\log p(x)$', fontsize=self._FS)
            if limitaxes:

                ax.axes.set_xlim3d(left=self._XYZ_LIM_LEFT, right=self._XYZ_LIM_RIGHT) 
                ax.axes.set_ylim3d(bottom=self._XYZ_LIM_LEFT, top=self._XYZ_LIM_RIGHT) 
                ax.axes.set_zlim3d(bottom=self._XYZ_LIM_LEFT, top=self._XYZ_LIM_RIGHT) 

        fig.tight_layout()        
        fig.subplots_adjust(wspace=0, hspace=0) 
        
class SixDimensionalNonSquareVisualizer(ThreeDimensionalVisualizerBase):
    _GRID_SIZE = 50
    _CONTOUR_LEVELS = 50

    def __init__(self, writer, x_train, num_elbo_samples, device):
        super().__init__(writer, x_train, device)

        self._num_elbo_samples = num_elbo_samples

        x1_lims = self._lims(self._x[:, 0])
        x2_lims = self._lims(self._x[:, 1])

        self._grid_x1, self._grid_x2 = torch.meshgrid((
            torch.linspace(*x1_lims, self._GRID_SIZE),
            torch.linspace(*x2_lims, self._GRID_SIZE)
        ))

        x1_x2 = torch.stack((self._grid_x1, self._grid_x2), dim=2).view(-1, 2)

        self._loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x1_x2.to(self._device)),
            batch_size=self._BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=False
        )
    def jacobian(self,y: torch.Tensor, x: torch.Tensor, create_graph=False):
        # -------------------------------------------------
        jac=torch.ones([y.shape[0],y.shape[1],x.shape[1]])
        for i in range(y.shape[1]):
            # for j in range(x.shape[1]):
                batched_grad = torch.ones_like(y.select(1,i))
                grad, = torch.autograd.grad(y.select(1,i),x,grad_outputs=batched_grad, is_grads_batched=False,  retain_graph=True, create_graph=False, allow_unused=False) 
                jac[:,i,:]=grad
        return jac
        
    def _plot_density(self, density, limitaxes=True):
        import matplotlib.gridspec as gridspec
    
        device = self._device
    
        # Select a subset of training points to visualize
        x = self._x.cpu()
        x = x[torch.randint(high=x.shape[0], size=(self._NUM_TRAIN_POINTS_TO_SHOW,))]
    
        # Generate different sets of numbers to visualize the embedded manifold in 6D
        low_dim_space = torch.linspace(-2.5, 2.5, self._NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW)
        numbers = [
            torch.cat([torch.randn(self._NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW, 1) for _ in range(6)], 1),
            *[
                torch.cat([low_dim_space[:, None] if i == j else torch.zeros(self._NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW, 1) for i in range(6)], 1)
                for j in range(6)
            ]
        ]
        numbers = [t.to(self._device) for t in numbers]
    
        # Labels for rows and columns
        row_labels = [r"$z_{all}$"] + [rf"$z_{i+1}$" for i in range(6)]
        col_labels = [
            r"$\mathbb{R}^{1,2,3}$", r"$\mathbb{R}^{2,3,4}$", r"$\mathbb{R}^{3,4,5}$",
            r"$\mathbb{R}^{4,5,6}$", r"$\mathbb{R}^{1,3,5}$", r"$\mathbb{R}^{2,4,6}$"
        ]
        dim_combinations = [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5), (0, 2, 4), (1, 3, 5)]
    
        # Compute metrics for \( J^TJ \)
        x_copy=self._x.cpu()
        x_copy.requires_grad = True
        low_dim_latent = density.extract_latent(x_copy, earliest_latent=True)
        x_ = density.fixed_sample(low_dim_latent)
        dxdz = self.jacobian(x_, low_dim_latent) #this is slow cannot stay here go gonna have to use jac in the other case
        g = torch.einsum("bki,bkj->bij", dxdz, dxdz)  # Compute J^TJ
        g_kk = torch.diagonal(g, dim1=-2, dim2=-1).mean(axis=0)
        g_kk = g_kk / torch.max(torch.abs(g_kk))
        v2 = g.mean(axis=0)
        v2 = v2 / torch.max(torch.abs(v2))
    
        # Compute cosine similarity
        norm_dxdz = torch.norm(dxdz, dim=2, keepdim=True)
        normalized_dxdz = dxdz / (norm_dxdz + 1e-8)
        cosine_similarity = torch.einsum("bki,bkj->bij", normalized_dxdz, normalized_dxdz)
        avg_cosine_similarity = torch.mean(cosine_similarity, dim=0)
    
        # Initialize results dictionary
        metrics_per_z = {
            "winding": [],
            "degree": [],
            "volume_distortion": [],
            "betti": [],
            "curvature_distortion": []
        }
        
        # Loop through rows of numbers (z_all + individual z_i)
        for row, numbers_row in enumerate(numbers):
            numbers_row.requires_grad=True
            # Step 1: Compute x_ from numbers_row
            embedded_manifold = density.fixed_sample(numbers_row)
            x_ = embedded_manifold  # x_ corresponds to numbers_row mapped to x
        
            # Step 2: Compute Jacobian (J) and metric tensor (g)
            # sparse_low_dim_latent_ = density.extract_latent(x_, earliest_latent=True)
            dxdz = self.jacobian(x_, numbers_row)  # Jacobian for the current latent sample
            g = torch.einsum("bki,bkj->bij", dxdz, dxdz)  # Compute J^TJ (metric tensor)
        
            # Step 3: Compute Topological Invariants
            try:
                # Winding Number
                winding = torch.det(dxdz @ dxdz.transpose(1, 2)).mean().detach().cpu().numpy()
                metrics_per_z["winding"].append(winding)
        
                # Degree
                degree = torch.sign(torch.det(dxdz)).sum().detach().cpu().numpy()
                metrics_per_z["degree"].append(degree)
        
                # Volume Distortion
                volume_distortion = torch.abs(torch.det(dxdz)).mean().detach().cpu().numpy()
                metrics_per_z["volume_distortion"].append(volume_distortion)
        
                # Betti Numbers
                eigenvalues = torch.linalg.eigvalsh(g @ g.transpose(1, 2))  # Eigenvalues of metric tensor
                betti = torch.sum(eigenvalues > 0, dim=1).float().mean().detach().cpu().numpy()
                metrics_per_z["betti"].append(betti)
        
                # Curvature Distortion
                curvature_distortion = torch.diagonal(g, dim1=-2, dim2=-1).sum(dim=-1).mean().detach().cpu().numpy()
                metrics_per_z["curvature_distortion"].append(curvature_distortion)
        
            except Exception as e:
                print(f"Error processing row {row} ({row_labels[row]}): {e}")
                # Append NaN for each metric in case of error
                metrics_per_z["winding"].append(np.nan)
                metrics_per_z["degree"].append(np.nan)
                metrics_per_z["volume_distortion"].append(np.nan)
                metrics_per_z["betti"].append(np.nan)
                metrics_per_z["curvature_distortion"].append(np.nan)


        fig = plt.figure(figsize=(28, 28))
        gs = gridspec.GridSpec(7, 7, width_ratios=[2, 4, 4, 4, 4, 4, 4])
    
 
    
        # Plot J (Jacobian components) below with labeled axes
        ax_j = plt.subplot(gs[0, 0])
        j_avg = dxdz.mean(axis=0).detach().cpu().numpy()
        im_j = ax_j.imshow(j_avg, cmap="RdBu", vmin=-1, vmax=1)

        # Add axis labels for x and z
        num_z = dxdz.size(1)  # Number of latent dimensions (z)
        num_x = dxdz.size(2)  # Number of data dimensions (x)
        ax_j.set_xticks(range(num_x))
        ax_j.set_yticks(range(num_z))
        ax_j.set_xticklabels([f"$x_{i+1}$" for i in range(num_x)], fontsize=10, rotation=45)
        ax_j.set_yticklabels([f"$z_{i+1}$" for i in range(num_z)], fontsize=10)
        ax_j.set_xlabel("Data Dimensions ($x$)", fontsize=12)
        ax_j.set_ylabel("Latent Dimensions ($z$)", fontsize=12)

        plt.colorbar(im_j, ax=ax_j, orientation="vertical", label=r"$J$")
        ax_j.set_title("Jacobian Components", fontsize=12)
        ax_j.axis("on")
    
        # Plot J^TJ below
        ax_jtj = plt.subplot(gs[1, 0])
        jtj_avg = g.mean(axis=0).detach().cpu().numpy()
        im_jtj = ax_jtj.imshow(jtj_avg, cmap="RdBu", vmin=-1, vmax=1)
        plt.colorbar(im_jtj, ax=ax_jtj, orientation="vertical", label=r"$J^TJ$")
        ax_jtj.set_title("Jacobian Metric", fontsize=12)
        ax_jtj.axis("off")
  
        # Define xtick_labels dynamically based on metrics length
        xtick_labels = [r"$z_{all}$"] + [f"$z_{i+1}$" for i in range(len(metrics_per_z["winding"]) - 1)]
        
        # Plot Winding Number
        ax_winding = plt.subplot(gs[2, 0])
        im_winding = ax_winding.imshow(
            np.array(metrics_per_z["winding"]).reshape(1, -1), cmap="RdBu", aspect="auto"
        )
        plt.colorbar(im_winding, ax=ax_winding, orientation="horizontal", label="Winding Number")
        ax_winding.set_title("Winding Number", fontsize=12)
        ax_winding.set_yticks([])
        ax_winding.set_xticks(range(len(metrics_per_z["winding"])))
        ax_winding.set_xticklabels(xtick_labels, rotation=45)
        
        # Topological Metric 2: Degree
        ax_degree = plt.subplot(gs[3, 0])
        im_degree = ax_degree.imshow(
            np.array(metrics_per_z["degree"]).reshape(1, -1), cmap="RdBu", aspect="auto"
        )  # Degree for each z_i
        plt.colorbar(im_degree, ax=ax_degree, orientation="horizontal", label="Degree")
        ax_degree.set_title("Degree", fontsize=12)
        ax_degree.set_yticks([])
        ax_degree.set_xticks(range(len(metrics_per_z["degree"])))
        ax_degree.set_xticklabels(xtick_labels, rotation=45)
        
        # Topological Metric 3: Volume Distortion
        ax_volume_distortion = plt.subplot(gs[4, 0])
        im_volume_distortion = ax_volume_distortion.imshow(
            np.array(metrics_per_z["volume_distortion"]).reshape(1, -1), cmap="RdBu", aspect="auto"
        )  # Volume distortion for each z_i
        plt.colorbar(im_volume_distortion, ax=ax_volume_distortion, orientation="horizontal", label="Volume Distortion")
        ax_volume_distortion.set_title("Volume Distortion", fontsize=12)
        ax_volume_distortion.set_yticks([])
        ax_volume_distortion.set_xticks(range(len(metrics_per_z["volume_distortion"])))
        ax_volume_distortion.set_xticklabels(xtick_labels, rotation=45)
        
        # Topological Metric 4: Betti number
        ax_betti = plt.subplot(gs[5, 0])
        im_betti = ax_betti.imshow(
            np.array(metrics_per_z["betti"]).reshape(1, -1), cmap="RdBu", aspect="auto"
        )  # Betti number for each z_i
        plt.colorbar(im_betti, ax=ax_betti, orientation="horizontal", label="Betti Number")
        ax_betti.set_title("Betti Number", fontsize=12)
        ax_betti.set_yticks([])
        ax_betti.set_xticks(range(len(metrics_per_z["betti"])))
        ax_betti.set_xticklabels(xtick_labels, rotation=45)
        
        # Topological Metric 4: Betti number
        ax_cd = plt.subplot(gs[6, 0])
        im_cd = ax_cd.imshow(
            np.array(metrics_per_z["curvature_distortion"]).reshape(1, -1), cmap="RdBu", aspect="auto"
        )  # Betti number for each z_i
        plt.colorbar(im_cd, ax=ax_cd, orientation="horizontal", label="curvature_distortion Number")
        ax_cd.set_title("curvature_distortion", fontsize=12)
        ax_cd.set_yticks([])
        ax_cd.set_xticks(range(len(metrics_per_z["curvature_distortion"])))
        ax_cd.set_xticklabels(xtick_labels, rotation=45)        


        # Add density plots to the rest of the grid
        for row in range(7):  # 7 rows for z_all and z_1 to z_6
            for col, dims in enumerate(dim_combinations):  # Columns based on dim_combinations
                ax = fig.add_subplot(gs[row, col + 1], projection='3d')
                ax.grid(False)
                ax.set_axis_off()
        
                # Add row and column labels
                if col == 0:
                    ax.text2D(0.5, 1.1, row_labels[row], fontsize=10, ha="center", va="center", transform=ax.transAxes, color="blue")
                if row == 0:
                    ax.text2D(0.5, 1.1, col_labels[col], fontsize=10, ha="center", va="center", transform=ax.transAxes, color="red")
        
                # Get the embedded manifold by sampling the current set of numbers
                embedded_manifold = density.fixed_sample(numbers[row])
                log_probs = density.elbo(
                    x=embedded_manifold,
                    add_reconstruction=False,
                    likelihood_wt=1.0,
                    visualization=True
                )["elbo"].squeeze()
        
                # Convert to numpy and normalize
                embedded_manifold = embedded_manifold.cpu().detach().numpy()
                log_probs = log_probs.cpu().detach().numpy()
        
                # Normalize log probabilities
                scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
                log_probs = scaler.fit_transform(log_probs.reshape(-1, 1)).ravel()
        
                # Scatter training points
                ax.scatter(
                    x[:, dims[0]].detach().numpy(),
                    x[:, dims[1]].detach().numpy(),
                    x[:, dims[2]].detach().numpy(),
                    c="k",
                    marker=".",
                    s=7,
                    linewidth=0.5,
                    alpha=0.3
                )
        
                # Scatter embedded manifold with log probabilities
                im = ax.scatter(
                    embedded_manifold[:, dims[0]],
                    embedded_manifold[:, dims[1]],
                    embedded_manifold[:, dims[2]],
                    c=log_probs,
                    cmap=self._CMAP,
                    marker="o",
                    s=40
                )
        
                # Add axis limits if enabled
                if limitaxes:
                    zoom_factor = 1.33
                    ax.set_xlim3d(left=self._XYZ_LIM_LEFT * zoom_factor, right=self._XYZ_LIM_RIGHT * zoom_factor)
                    ax.set_ylim3d(bottom=self._XYZ_LIM_LEFT * zoom_factor, top=self._XYZ_LIM_RIGHT * zoom_factor)
                    ax.set_zlim3d(bottom=self._XYZ_LIM_LEFT * zoom_factor, top=self._XYZ_LIM_RIGHT * zoom_factor)
        
        # Adjust layout and add a colorbar for the entire figure
        cbaxes = fig.add_axes([1, 0.15, 0.015, 0.7])  # Position for the colorbar
        axcb = fig.colorbar(im, cax=cbaxes, shrink=0.6)
        axcb.set_label(r'$\log p(x)$', fontsize=self._FS)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)
        

        
class FourDimensionalNonSquareVisualizer(ThreeDimensionalVisualizerBase):
    _GRID_SIZE = 50
    _CONTOUR_LEVELS = 50

    def __init__(self, writer, x_train, num_elbo_samples, device):
        super().__init__(writer, x_train, device)

        self._num_elbo_samples = num_elbo_samples

        x1_lims = self._lims(self._x[:, 0])
        x2_lims = self._lims(self._x[:, 1])

        self._grid_x1, self._grid_x2 = torch.meshgrid((
            torch.linspace(*x1_lims, self._GRID_SIZE),
            torch.linspace(*x2_lims, self._GRID_SIZE)
        ))

        x1_x2 = torch.stack((self._grid_x1, self._grid_x2), dim=2).view(-1, 2)

        self._loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x1_x2.to(self._device)),
            batch_size=self._BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=False
        )

        
    def _plot_density(self, density, limitaxes=True):
        device = self._device
    
        # Select a subset of training points to visualize
        x = self._x.cpu()
        x = x[torch.randint(high=x.shape[0], size=(self._NUM_TRAIN_POINTS_TO_SHOW,))]
    
        # Generate different sets of numbers to visualize the embedded manifold
        low_dim_space = torch.linspace(-2.5, 2.5, self._NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW)
        numbers = [
            torch.cat([torch.randn(self._NUM_SAMPLE_POINTS_TO_SHOW, 1) for _ in range(4)], 1),
            torch.cat([low_dim_space[:, None] if i == 0 else torch.zeros(self._NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW, 1) for i in range(4)], 1),
            torch.cat([low_dim_space[:, None] if i == 1 else torch.zeros(self._NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW, 1) for i in range(4)], 1),
            torch.cat([low_dim_space[:, None] if i == 2 else torch.zeros(self._NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW, 1) for i in range(4)], 1),
            torch.cat([low_dim_space[:, None] if i == 3 else torch.zeros(self._NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW, 1) for i in range(4)], 1),
        ]
        numbers = [t.to(self._device) for t in numbers]
    
        # Define labels for each row and column
        row_labels = [r"$z_{all}$", r"$z_1$", r"$z_2$", r"$z_3$", r"$z_4$"]
        col_labels = [r"$\mathbb{R}^{1,2,3}$", r"$\mathbb{R}^{2,3,4}$", r"$\mathbb{R}^{1,2,4}$", r"$\mathbb{R}^{1,3,4}$"]
        dim_combinations = [(0, 1, 2), (1, 2, 3), (0, 1, 3), (0, 2, 3)]
    
        # Create the figure with a larger size to accommodate the grid
        fig = plt.figure(figsize=(16, 20))  # Increase figure size for better clarity
    
        # Plotting loop for a 5x4 grid
        for row in range(len(numbers)):
            for col, dims in enumerate(dim_combinations):
                ax = fig.add_subplot(5, 4, row * 4 + col + 1, projection='3d')
                ax.grid(False)
                ax.set_axis_off()
    
                # Set the row and column labels
                if col == 0:
                    ax.text(-2, 1, 0, f"{row_labels[row]}", fontsize=self._FS, color="blue")
                if row == 0:
                    ax.text(0, 2, 0, f"{col_labels[col]}", fontsize=self._FS, color="red")
    
                # Get the embedded manifold by sampling the current set of numbers
                embedded_manifold = density.fixed_sample(numbers[row])
                log_probs = density.elbo(
                    x=embedded_manifold,
                    add_reconstruction=False,
                    likelihood_wt=1.0,
                    visualization=True
                )["elbo"].squeeze()
    
                # Convert to numpy and normalize
                embedded_manifold = embedded_manifold.cpu().detach().numpy()
                log_probs = log_probs.cpu().detach().numpy()
    
                scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
                log_probs = scaler.fit_transform(log_probs.reshape(-1, 1)).ravel()
    
                # Plot the points in the current set of dimensions
                ax.scatter(x[:, dims[0]], x[:, dims[1]], x[:, dims[2]], c="k", marker=".", s=7, linewidth=0.5, alpha=0.3)
                im = ax.scatter(embedded_manifold[:, dims[0]], embedded_manifold[:, dims[1]], embedded_manifold[:, dims[2]], c=log_probs, cmap=self._CMAP, marker="o", s=40)
    
                if limitaxes:
                    # Zoom out by expanding the axis limits by approximately 1/3
                    zoom_factor = 1.33
                    ax.set_xlim3d(left=self._XYZ_LIM_LEFT * zoom_factor, right=self._XYZ_LIM_RIGHT * zoom_factor)
                    ax.set_ylim3d(bottom=self._XYZ_LIM_LEFT * zoom_factor, top=self._XYZ_LIM_RIGHT * zoom_factor)
                    ax.set_zlim3d(bottom=self._XYZ_LIM_LEFT * zoom_factor, top=self._XYZ_LIM_RIGHT * zoom_factor)
    
        # Add colorbar for the entire figure
        cbaxes = fig.add_axes([1, 0.15, 0.015, 0.7])
        axcb = fig.colorbar(im, cax=cbaxes, shrink=0.6)
        axcb.set_label(r'$\log p(x)$', fontsize=self._FS)
        fig.tight_layout()        
        fig.subplots_adjust(wspace=0, hspace=0)

    




 
