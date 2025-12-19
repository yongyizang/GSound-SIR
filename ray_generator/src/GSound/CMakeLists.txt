project( gsound )

file(GLOB_RECURSE SOURCES gsound/*.cpp)

if(GSOUND_USE_OPTIX)
    # CUDA Source - this will be compiled by NVCC
    set(CUDA_SOURCES optix/kernels.cu)
    
    # Host Source
    file(GLOB_RECURSE OPTIX_HOST_SOURCES optix/*.cpp)
    
    # cuda_compile_ptx is not a built-in CMake command in newer CUDAToolkit.
    # We should use standard add_library or custom command. 
    # For now, let's use a custom command to generate PTX.
    
    # Simple rule to compile .cu to .ptx
    set(PTX_FILES "")
    foreach(cu_file ${CUDA_SOURCES})
        get_filename_component(base_name ${cu_file} NAME_WE)
        set(ptx_file "${CMAKE_CURRENT_BINARY_DIR}/${base_name}.ptx")
        list(APPEND PTX_FILES ${ptx_file})
        
        add_custom_command(
            OUTPUT ${ptx_file}
            COMMAND ${CMAKE_CUDA_COMPILER} 
                    -ptx 
                    ${CMAKE_CURRENT_SOURCE_DIR}/${cu_file} 
                    -o ${ptx_file}
                    --include-path ${OPTIX_INCLUDE_DIR_FOUND}
            DEPENDS ${cu_file}
            COMMENT "Compiling ${cu_file} to PTX"
        )
    endforeach()
    
    # We need to make sure the PTX is generated before compiling host code that might use it
    # add_custom_target(generate_ptx DEPENDS ${PTX_FILES})
    
    # Add sources
    list(APPEND SOURCES ${OPTIX_HOST_SOURCES})
    
    # Expose the PTX path to C++ code via a definition
    # For simplicity, we assume one kernel file for now
    add_definitions(-DOPTIX_KERNEL_PTX_FILE="${CMAKE_CURRENT_BINARY_DIR}/kernels.ptx")
endif()

set( SOURCEFILES ${SOURCES} )

add_library( gsound STATIC ${SOURCEFILES})

if(GSOUND_USE_OPTIX)
    # The 'generate_ptx' custom target might not propagate dependency correctly for a static library source
    # Instead, we just ensure it runs.
    add_custom_target(generate_ptx DEPENDS ${PTX_FILES})
    add_dependencies(gsound generate_ptx)
endif()

set( EXTERNAL_LIBS
		om-sound
		om-resources
		om-bvh
		om-framework
		fftw3_threads
		fftw3f
		fftw3
)

if(GSOUND_USE_OPTIX)
    list(APPEND EXTERNAL_LIBS CUDA::cudart CUDA::cuda_driver)
endif()

target_link_libraries( gsound ${EXTERNAL_LIBS} )


