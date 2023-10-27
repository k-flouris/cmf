import os

# from collections import defaultdict

import numpy as np

import torch
import torch.utils.data
import torchvision.utils

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (4,4)
from matplotlib.collections import LineCollection

import tqdm

from .metrics import metrics

from scipy.special import i0
from scipy.stats import vonmises
from scipy.stats import gaussian_kde
from sklearn import preprocessing


import random
import json
# import seaborn
try: from .kf_fid_score import fid_from_samples, m_s_from_samples
except: None
# TODO: Make return a matplotlib figure instead. Writing can be done outside.


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


# NB test metric plots:
# ---------------------------------------------------------------------------------- 
# ---------------------------------------------------------------------------------- 
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
        #  NB. Here uncomment and comment at will, various analysis plots
        density.eval()
        metric_sort_index=self.calculate_metric_and_sort_ld(density)
        self.calculate_metric_and_sort_ld(density)
        self.plot_metric_components(metric_sort_index,density, epoch, write_folder, fixed_noise, extent, labels)
        self.plot_fid_for_effective_z(metric_sort_index,density, epoch, write_folder, fixed_noise, extent, labels)
        self.plot_pairwise_dimension_comparison(metric_sort_index,density, epoch, write_folder, fixed_noise, extent, labels)
       
        # self.plot_samples_prominent_z_indivitual(metric_sort_index,density, epoch, write_folder, fixed_noise, extent, labels, bs=15, num_of_dims=10)
        # self.plot_samples_prominent_z_combined(metric_sort_index,density, epoch, write_folder, fixed_noise, extent, labels, num_of_dims=5)
        # self.plot_samples_prominent_z_cumulative(metric_sort_index,density, epoch, write_folder, fixed_noise, extent, labels, bs=15,num_of_dims=10)
        # self.plot_samples_prominent_z_hierarchical(metric_sort_index,density, epoch, write_folder, fixed_noise, extent, labels, num_of_dims=10)
        
        # NEEDS FIXING:
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

    def plot_metric_components(self,sort_index, density, epoch, write_folder=None, fixed_noise=None, extent=None, labels=None, num_of_dims=5, bs=16):
        # try:
        #     import tensorflow as tf
        # except: 
        #     print("No module tensorflow, exiting <<plot_fid_for_effective_z>> ")
        #     return None
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
        x_= density.fixed_sample(low_dim_latent) 
        dxdz = self.jacobian(x_,low_dim_latent)
        g=torch.einsum("bki,bkj->bij",dxdz,dxdz)
        g_kk=torch.diagonal(g,  dim1=-2, dim2=-1)
        print(g.shape)

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
        

        fig, (ax1, ax2) = plt.subplots(2, 1,figsize=plt.figaspect(2))

        im1=ax1.imshow(v1, cmap=cmap, vmin=-1, vmax=1)
        cbar=fig.colorbar(im1, ax=ax1, orientation='horizontal')
        cbar.set_label(r"$G_{kk}$")

        # Plot the second image in the bottom subplot
        im2=ax2.imshow(average_cosine_similarity[0], cmap=cmap, vmin=-0.5, vmax=0.5, aspect='auto')
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
        # now with train data maybe this is gonna work see to use them as below
        # jtj, _ = density._get_full_jac_transpose_jac(low_dim_latent, False)
        # log_jac_jac_t = density.pullback_log_jac_jac_transpose(low_dim_latent)
        g_kk_sort_index = np.argsort(torch.abs(g_kk.mean(axis=0)))

        return g_kk_sort_index
    
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
            for (x_batch, labels) in self._test_loader: # iterate batches
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
                # if test_i == 2:break
            

            recon=torch.nn.functional.mse_loss(x_all,xhat_all)
            recons.append(recon.cpu().detach().numpy())
            print(xhat_all.shape,x_all.shape)
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
    def plot_samples_prominent_z_indivitual(self,sort_index, density, epoch, write_folder=None, fixed_noise=None,  extent=None, labels=None,bs=30, num_of_dims=5, random_seed=int(1454)):
        density.eval()
        x=self._x[-bs:] 
        low_dim_latent=density.extract_latent(x,earliest_latent=True)
        low_dim_latent=low_dim_latent.to(self._device)      
        ld=low_dim_latent.shape[1]
        self.set_seed(s=random_seed)
        low_dim_latent=torch.randn(bs,ld)
        low_dim_latent=low_dim_latent.to(self._device)
        reverse_sort_index = sorted(range(ld), key=sort_index.__getitem__)
           
        latent_dims_index=np.arange(ld)
        prominent_subgroups_index = np.array_split(latent_dims_index, num_of_dims)
    
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
        low_dim_latent=density.extract_latent(x,earliest_latent=True)
        low_dim_latent=low_dim_latent.to(self._device)      
        ld=low_dim_latent.shape[1]
        self.set_seed(s=14545)
        low_dim_latent=torch.randn(bs,ld)
        # low_dim_latent=torch.linspace(-3,3,bs).unsqueeze(-1).repeat(1,ld)

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
        # low_dim_latent_=low_dim_latent_[:,sort_index]
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
 
            
    # @torch.no_grad()
    # def plot_samples_for_effective_z(self,sort_index, density, epoch, write_folder=None, fixed_noise=None,num_of_dims=5, random_seed=int(5545), extent=None, labels=None,bs=10):
    #     density.eval()
    #     x=self._x[-bs:] 
    #     xdim=x.shape[1]
    #     ydim=x.shape[2]
    #     low_dim_latent=density.extract_latent(x,earliest_latent=True)
    #     low_dim_latent=low_dim_latent.to(self._device)      
    #     ld=low_dim_latent.shape[1]

    #     set_seed(random_seed)
    #     low_dim_latent=torch.randn(bs,ld)
    #     low_dim_latent=low_dim_latent.to(self._device)
    #     # TODO why is this memory crushing with more x etc?
    #     reverse_sort_index = sorted(range(ld), key=sort_index.__getitem__)

    #     mult=int(ld/num_of_dims)
    #     dimensions=list(range(num_of_dims+1))
    #     dimensions=[item * mult for item in dimensions]
    #     print(dimensions)

    #     for i in dimensions:
    #         if i==0: i=1
    #         low_dim_latent_=low_dim_latent[:,sort_index]#largest is last
    #         low_dim_latent_[:,:-i]=torch.zeros(ld-i)
    #         # low_dim_latent_[:,i:]=torch.zeros(ld-i)

    #         low_dim_latent_=low_dim_latent_[:,reverse_sort_index]#put them back to their location
    #         imgs = density.fixed_sample(low_dim_latent_)
    #         if i==1:
    #             images=imgs
    #         else:
    #             images=torch.concat((images,imgs),axis=0)

    #     imgs=images
    #     num_rows = bs

    #     grid = torchvision.utils.make_grid(
    #         imgs, nrow=num_rows, pad_value=1,
    #         normalize=True, scale_each=True
    #     )

    #     grid_permuted = grid.permute((1,2,0))
    #     axes=plt.imshow(grid_permuted.detach().cpu().numpy(), extent=extent)

    #     ticksy=dimensions
    #     ticksx=range(1,bs+1,1)
    #     xx = [i * xdim for i in ticksx]
    #     yy = [i * ydim/mult for i in ticksy]
    #     plt.xlabel('Samples')
    #     plt.ylabel('d')
    #     plt.xticks(xx, ticksx)
    #     plt.yticks(yy, ticksy)

    #     if write_folder:
    #         savedir = os.path.join(write_folder, "test_metric")
    #         if not os.path.exists(savedir):
    #             os.makedirs(savedir)
    #         plt.savefig(os.path.join(savedir, "samples_for_effective_ld.pdf"))
    #     else:
    #         self._writer.write_image("samples_for_effective_ld", grid, global_step=epoch)  

            
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

    # kf:
    def jacobian(self,y: torch.Tensor, x: torch.Tensor, create_graph=False):
        # -------------------------------------------------
        jac=torch.ones([y.shape[0],y.shape[1],x.shape[1]])
        for i in range(y.shape[1]):
            # for j in range(x.shape[1]):
                batched_grad = torch.ones_like(y.select(1,i))
                grad, = torch.autograd.grad(y.select(1,i),x,grad_outputs=batched_grad, is_grads_batched=False,  retain_graph=True, create_graph=False, allow_unused=False) 
                jac[:,i,:]=grad
        return jac
    
# metricplots-end
# ---------------------------------------------------------------------------------- 
# ---------------------------------------------------------------------------------- 
# ---------------------------------------------------------------------------------- 
      
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
        
        # Do KS statistics
        from scipy.stats import kstest
        
        embedded_manifold = density.fixed_sample(numbers[0])
        log_probs = density.elbo(
            x=embedded_manifold,
            add_reconstruction=False,
            likelihood_wt=1.,
            visualization=True
        )["elbo"].squeeze()
        
        L=log_probs.detach().numpy()
        x_sorted = np.sort(x[:,0].detach().numpy())
        cdf_x = x_sorted

        e_sorted = np.sort(embedded_manifold[:,0].detach().numpy())
        cdf_e = e_sorted

        # Calculate the KS statistic
        # ks_statistic_raw = np.max(np.abs(cdf_x - np.sort(L)))
        ks_statistics, ks_p_value = kstest(cdf_e,cdf_x)

        print("KS Statistic:", ks_statistics)
        # print("KS Statistic_raw:", ks_statistic_raw)
        print("KS p-value:", ks_p_value)
        
class ThreeDimensionalVisualizerBase(DensityVisualizer):
        _NUM_TRAIN_POINTS_TO_SHOW = 500
        _NUM_SAMPLE_POINTS_TO_SHOW = 500
        _NUM_SAMPLE_POINTS_INLOWDIM_TO_SHOW = 100
    
        _BATCH_SIZE = 1000
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

        fig = plt.figure(figsize=plt.figaspect(0.33))
        for i in range(len(numbers)):
            ax = fig.add_subplot(1, 4, int(i+1), projection='3d')
            ax.grid(False)
            ax.axis(False)
            ax.text(-2,1,0, str(labels[i]), fontsize=self._FS )
            embedded_manifold = density.fixed_sample(numbers[i])
            log_probs = density.elbo(
                x=embedded_manifold,
                add_reconstruction=False,
                likelihood_wt=1.,
                visualization=True
            )["elbo"].squeeze()
            embedded_manifold=embedded_manifold.cpu().detach().numpy()
            log_probs=log_probs.cpu().detach().numpy()

            scaler=preprocessing.MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(log_probs.reshape(-1, 1))
            log_probs=scaler.transform(log_probs.reshape(-1, 1)).ravel()

            
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], c="k", marker=".", s=7, linewidth=0.5, alpha=0.3)
            im=ax.scatter(embedded_manifold[:, 0], embedded_manifold[:, 1],embedded_manifold[:, 2], c=log_probs, cmap=self._CMAP, marker="o", s=40)
            ax.axes.set_xlim3d(left=self._XYZ_LIM_LEFT, right=self._XYZ_LIM_RIGHT) 
            ax.axes.set_ylim3d(bottom=self._XYZ_LIM_LEFT, top=self._XYZ_LIM_RIGHT) 
            ax.axes.set_zlim3d(bottom=self._XYZ_LIM_LEFT, top=self._XYZ_LIM_RIGHT) 

        cbaxes = fig.add_axes([1, 0.15, 0.015, 0.7])
        axcb=fig.colorbar(im,cax=cbaxes, shrink=0.6)
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
            embedded_manifold=embedded_manifold.cpu().detach().numpy()
            log_probs=log_probs.cpu().detach().numpy()
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], c="k", marker=".", s=7, linewidth=0.5, alpha=0.2)
            im=ax.scatter(embedded_manifold[:, 0], embedded_manifold[:, 1],embedded_manifold[:, 2], c=log_probs, cmap=self._CMAP, marker="o", s=7)
            axcb= fig.colorbar(im,extend="both", shrink=0.8)
            axcb.set_label(r'$\log p(x)$', fontsize=self._FS)
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

            embedded_manifold=embedded_manifold.cpu().detach().numpy()
            log_probs=log_probs.cpu().detach().numpy()

            ax.scatter(x[:, 0], x[:, 1], x[:, 2], c="k", marker=".", s=7, linewidth=0.5, alpha=0.2)
            im=ax.scatter(embedded_manifold[:, 0], embedded_manifold[:, 1],embedded_manifold[:, 2], c=log_probs, cmap=self._CMAP, marker="o", s=7)
            axcb= fig.colorbar(im,extend="both", shrink=0.8)
            axcb.set_label(r'$\log p(x)$', fontsize=self._FS)
            ax.axes.set_xlim3d(left=self._XYZ_LIM_LEFT, right=self._XYZ_LIM_RIGHT) 
            ax.axes.set_ylim3d(bottom=self._XYZ_LIM_LEFT, top=self._XYZ_LIM_RIGHT) 
            ax.axes.set_zlim3d(bottom=self._XYZ_LIM_LEFT, top=self._XYZ_LIM_RIGHT) 

        fig.tight_layout()        
        fig.subplots_adjust(wspace=0, hspace=0) 
        
            
    

        




 