#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
void loc_debris_hydforce_cuda(at::Tensor dx,
                at::Tensor dt_g, 
                at::Tensor maskIndex,
                at::Tensor num_x_device,
                at::Tensor num_y_device,
                at::Tensor zb_device,
                at::Tensor h_device,
                at::Tensor z_device,
                at::Tensor qx_device,
                at::Tensor qy_device, 
                at::Tensor g_gne_device,
                at::Tensor g_gbar_device,
                at::Tensor g_glay_device,
                at::Tensor g_gR_device, 
                at::Tensor g_gp_device,
                at::Tensor g_gt_device,
                at::Tensor g_gm_device,
                at::Tensor g_gz_device,
                at::Tensor g_gx_device,
                at::Tensor g_gy_device,
                at::Tensor neloc_x_device,
                at::Tensor neloc_y_device,
                at::Tensor mu,
                at::Tensor g_properties,
                at::Tensor g_grho_device,
                at::Tensor g_gvx_device,
                at::Tensor g_gvy_device,	
                at::Tensor g_gfx_device,
                at::Tensor g_gfy_device,
                at::Tensor cell_f_xP_device,
                at::Tensor cell_f_xN_device,
                at::Tensor cell_f_yP_device,
                at::Tensor cell_f_yN_device,
                at::Tensor tau_wgx_device, 
                at::Tensor tau_wgy_device, 
                at::Tensor g_gI_device,
                at::Tensor g_gax_device,
                at::Tensor g_gay_device,
                at::Tensor g_gang_device,
                at::Tensor g_gangv_device,
                at::Tensor g_ganga_device);

// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void locDebrisAndcalHydforce(at::Tensor dx,
                at::Tensor dt_g, 
                at::Tensor maskIndex,
                at::Tensor num_x_device,
                at::Tensor num_y_device,
                at::Tensor zb_device,
                at::Tensor h_device,
                at::Tensor z_device,
                at::Tensor qx_device,
                at::Tensor qy_device, 
                at::Tensor g_gne_device,
                at::Tensor g_gbar_device,
                at::Tensor g_glay_device,
                at::Tensor g_gR_device, 
                at::Tensor g_gp_device,
                at::Tensor g_gt_device,
                at::Tensor g_gm_device,
                at::Tensor g_gz_device,
                at::Tensor g_gx_device,
                at::Tensor g_gy_device,
                at::Tensor neloc_x_device,
                at::Tensor neloc_y_device,
                at::Tensor mu,
                at::Tensor g_properties,
                at::Tensor g_grho_device,
                at::Tensor g_gvx_device,
                at::Tensor g_gvy_device,	
                at::Tensor g_gfx_device,
                at::Tensor g_gfy_device,
                at::Tensor cell_f_xP_device,
                at::Tensor cell_f_xN_device,
                at::Tensor cell_f_yP_device,
                at::Tensor cell_f_yN_device,
                at::Tensor tau_wgx_device, 
                at::Tensor tau_wgy_device, 
                at::Tensor g_gI_device,
                at::Tensor g_gax_device,
                at::Tensor g_gay_device,
                at::Tensor g_gang_device,
                at::Tensor g_gangv_device,
                at::Tensor g_ganga_device) {
                    
    CHECK_INPUT(dx);
    CHECK_INPUT(dt_g);
    CHECK_INPUT(maskIndex);
    CHECK_INPUT(num_x_device);
    CHECK_INPUT(num_y_device);
    CHECK_INPUT(zb_device);
    CHECK_INPUT(h_device);
    CHECK_INPUT(z_device);
    CHECK_INPUT(qx_device);
    CHECK_INPUT(qy_device);
    CHECK_INPUT(g_gne_device);
    CHECK_INPUT(g_gbar_device);
    CHECK_INPUT(g_glay_device);
    CHECK_INPUT(g_gR_device);
    CHECK_INPUT(g_gp_device);
    CHECK_INPUT(g_gt_device);
    CHECK_INPUT(g_gm_device);
    CHECK_INPUT(g_gz_device);
    CHECK_INPUT(g_gx_device);
    CHECK_INPUT(g_gy_device);
    CHECK_INPUT(neloc_x_device);
    CHECK_INPUT(neloc_y_device);
    CHECK_INPUT(mu);
    CHECK_INPUT(g_properties);
    CHECK_INPUT(g_grho_device);
    CHECK_INPUT(g_gvx_device);
    CHECK_INPUT(g_gvy_device);
    CHECK_INPUT(g_gfx_device);
    CHECK_INPUT(g_gfy_device);
    CHECK_INPUT(cell_f_xP_device);
    CHECK_INPUT(cell_f_xN_device);
    CHECK_INPUT(cell_f_yP_device);
    CHECK_INPUT(cell_f_yN_device);
    CHECK_INPUT(tau_wgx_device);
    CHECK_INPUT(tau_wgy_device);
    CHECK_INPUT(g_gI_device);
    CHECK_INPUT(g_gax_device);
    CHECK_INPUT(g_gay_device);
    CHECK_INPUT(g_gang_device);
    CHECK_INPUT(g_gangv_device);
    CHECK_INPUT(g_ganga_device);	

    loc_debris_hydforce_cuda(dx,
                dt_g, 
                maskIndex,
                num_x_device,
                num_y_device,
                zb_device,
                h_device,
                z_device,
                qx_device,
                qy_device, 
                g_gne_device,
                g_gbar_device,
                g_glay_device,
                g_gR_device, 
                g_gp_device,
                g_gt_device,
                g_gm_device,
                g_gz_device,
                g_gx_device,
                g_gy_device,
                neloc_x_device,
                neloc_y_device,
                mu,
                g_properties,
                g_grho_device,
                g_gvx_device,
                g_gvy_device,	
                g_gfx_device,
                g_gfy_device,
                cell_f_xP_device,
                cell_f_xN_device,
                cell_f_yP_device,
                cell_f_yN_device,
                tau_wgx_device, 
                tau_wgy_device, 
                g_gI_device,
                g_gax_device,
                g_gay_device,
                g_gang_device,
                g_gangv_device,
                g_ganga_device);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update", &locDebrisAndcalHydforce,
        "calculate interaction between debris and walls, CUDA version");
}