from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_FLAGS = []

ext_modules = [
    CUDAExtension('fluxCalculation', [
        'fluxCal_interface.cpp',
        'fluxCal_Kernel.cu',
    ]),
    CUDAExtension('euler_update', [
        'euler_update.cpp',
        'euler_update_Kernel.cu',
    ]),
    # CUDAExtension('fluxCalculation_2', [
    #     'fluxCalculation_2_interface.cpp',
    #     'fluxCalculation_2_Kernel.cu',
    # ]),
    CUDAExtension('fluxCalculation_1stOrder_Chen', [
        'fluxCal_1stOrder_interface_Chen.cpp',
        'fluxCal_1stOrder_Kernel_Chen.cu',
    ]),
    CUDAExtension('fluxCalculation_1stOrder_xilin_SMR', [
        'fluxCal_1stOrder_interface_xilin_SMR.cpp',
        'fluxCal_1stOrder_Kernel_xilin_SMR.cu',
    ]),
    CUDAExtension('fluxCal_1st_Jiaheng_surface', [
        'fluxCal_1st_Jiaheng_surface.cpp',
        'fluxCal_1st_Jiaheng_surface_Kernel.cu',
    ]),
    CUDAExtension('fluxCalculation_jh_modified_surface', [
        'fluxCal_jh_modified_surface.cpp',
        'fluxCal_jh_modified_surface_Kernel.cu',
    ]),
    CUDAExtension('fluxCalculation_1stOrder_Hou', [
        'fluxCal_1st_interface_Hou.cpp',
        'fluxCal_1st_Kernel_Hou.cu',
    ]),
    CUDAExtension('fluxCalculation_2ndOrder_jh', [
        'fluxCal_2ndOrder_jh_interface.cpp',
        'fluxCal_2ndOrder_jh_Kernel.cu',
    ]),
    CUDAExtension('fluxCalculation_2ndOrder_jh_modified', [
        'fluxCal_2ndOrder_jh_modified.cpp',
        'fluxCal_2ndOrder_jh_modified_Kernel.cu',
    ]),
    CUDAExtension('fluxCal_2ndOrder_jh_improved', [
        'fluxCal_2ndOrder_jh_improved.cpp',
        'flucCal_2ndOrder_jh_improved_Kernel.cu',
    ]),
    CUDAExtension('fluxCal_2ndOrder_chen_improved', [
        'fluxCal_2ndOrder_chen_improved.cpp',
        'flucCal_2ndOrder_chen_improved_Kernel.cu',
    ]),
    CUDAExtension('fluxCalculation_2ndOrder_chen', [
        'fluxCal_2ndOrder_chen.cpp',
        'fluxCal_2ndOrder_Chen_Kernel.cu',
    ]),
    # CUDAExtension('fluxCalculation_1stOrder_2', [
    #     'fluxCal_1st_2_interface.cpp',
    #     'fluxCal_1st_2_Kernel.cu',
    # ]),
    CUDAExtension('frictionCalculation', [
        'friction_interface.cpp',
        'frictionCUDA_Kernel.cu',
    ]),
    CUDAExtension('frictionCalculation_implicit', [
        'friction_implicit_interface.cpp',
        'friction_implicit_Kernel.cu',
    ]),
    CUDAExtension('friction_implicit_andUpdate_jh', [
        'friction_implicit_andUpdate_jh_interface.cpp',
        'friction_implicit_andUpdate_jh_Kernel.cu',
    ]),
    CUDAExtension('stormSource', [
        'stormSource.cpp',
        'stormSource_kernel.cu',
    ]),
    CUDAExtension('infiltrationCalculation', [
        'infiltration_interface.cpp',
        'infiltrationCUDA_Kernel.cu',
    ]),
    CUDAExtension('station_PrecipitationCalculation', [
        'stationPrecipitation_interface.cpp',
        'stationPrecipitation_Kernel.cu',
    ]),
    CUDAExtension('timeControl', [
        'timeControl.cpp',
        'timeControl_Kernel.cu',
    ]),
    CUDAExtension('fluxMask', [
        'fluxMaskGenerator.cpp',
        'fluxMaskGenerator_Kernel.cu',
    ]),

    # Wash-off model
    CUDAExtension('parInit', [
        'parInit_interface.cpp',
        'parInit_Kernel.cu',
    ]),
    CUDAExtension('multiParInit', [
        'multiParInit_interface.cpp',
        'multiParInit_Kernel.cu',
    ]),
    CUDAExtension('washoff', [
        'washoff_interface.cpp',
        'washoff_Kernel.cu',
    ]),
    CUDAExtension('transport', [
        'transport_interface.cpp',
        'transport_Kernel.cu',
    ]),
    CUDAExtension('update_after_transport', [
        'update_after_transport_interface.cpp',
        'update_after_transport_Kernel.cu',
    ]),
    CUDAExtension('multiUpdateAfterTransport', [
        'multiUpdateAfterTransport_interface.cpp',
        'multiUpdateAfterTransport_Kernel.cu',
    ]),
    CUDAExtension('update_after_transport', [
        'update_after_transport_interface.cpp',
        'update_after_transport_Kernel.cu',
    ]),
    CUDAExtension('update_after_washoff', [
        'update_after_washoff_interface.cpp',
        'update_after_washoff_Kernel.cu',
    ]),
    CUDAExtension('assign_particles', [
        'assignParticles_interface.cpp',
        'assignParticles_Kernel.cu',
    ]),
    CUDAExtension('multiUpdateAfterWashoff', [
        'multiUpdateAfterWashoff_interface.cpp',
        'multiUpdateAfterWashoff_Kernel.cu',
    ]),
    # DEM module
    CUDAExtension('elementLocation', [
        'element_location_update_interface.cpp',
        'element_location_update_Kernel.cu',
    ]),
    CUDAExtension('interactGrains', [
        'interact_grains_interface.cpp',
        'interact_grains_Kernel.cu',
    ]),
    CUDAExtension('interactWalls', [
        'interact_walls_interface.cpp',
        'interact_walls_Kernel.cu',
    ]),
]

# INSTALL_REQUIREMENTS = ['numpy', 'torch', 'torchvision', 'scikit-image', 'tqdm', 'imageio']

setup(
    description='PyTorch implementation of "HiPIMS"',
    author='Jiaheng Zhao',
    author_email='j.zhao@lboro.ac.uk',
    license='in CopyRight: in-house code',
    version='1.1.0',
    name='hipims',
    extra_compile_args={
        'cxx': ['-std=c++11', '-O2', '-Wall'],
        'nvcc': [
            '-std=c++11', '--expt-extended-lambda', '--use_fast_math',
            '-Xcompiler', '-Wall', '-gencode=arch=compute_60,code=sm_60',
            '-gencode=arch=compute_61,code=sm_61',
            '-gencode=arch=compute_70,code=sm_70',
            '-gencode=arch=compute_72,code=sm_72',
            '-gencode=arch=compute_75,code=sm_75',
            '-gencode=arch=compute_75,code=compute_75'
        ],
    },
    # packages=['hipims'],
    #     install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    })
