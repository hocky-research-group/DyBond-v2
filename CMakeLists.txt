# Template CMakeLists.txt for plugins/components.
CMAKE_MINIMUM_REQUIRED(VERSION 3.9 FATAL_ERROR)
project(dybond LANGUAGES C CXX)
#cmake_policy(SET CMP0074 NEW)
set(COMPONENT_NAME dybond)

# Find the installed HOOMD
find_package(HOOMD REQUIRED)

# Manually verify that the version is >= 4.0.0 (to allow major versions 4, 5, 6, 7+)
if(HOOMD_VERSION VERSION_LESS 4.0.0)
    message(FATAL_ERROR "HOOMD-blue version 4.0.0 or newer is required. Found version: ${HOOMD_VERSION}")
endif()

message(STATUS "Found HOOMD ${HOOMD_VERSION}: ${HOOMD_INSTALL_PREFIX}/${PYTHON_SITE_INSTALL_DIR}")

# Force installation to the HOOMD installation location
set(CMAKE_INSTALL_PREFIX ${HOOMD_INSTALL_PREFIX} CACHE PATH "Installation prefix" FORCE)

# Enable compiler warnings on gcc and clang (common compilers used by developers)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wconversion -Wno-sign-conversion -Wno-unknown-pragmas -Wno-deprecated-declarations -Wno-unused-result")
endif()

# Specify any C++ sources
set(_${COMPONENT_NAME}_sources
    src/cpp/DyBondBase.cc
    src/cpp/DyBond.cc
    src/cpp/DyBondUpdater.cc
    src/cpp/module.cc
    )

# Specify any CUDA sources
set(_${COMPONENT_NAME}_cu_sources
    src/cpp/DyBond.cu
    )

if (ENABLE_HIP)
set(_cuda_sources ${_${COMPONENT_NAME}_cu_sources})
endif (ENABLE_HIP)

pybind11_add_module(_${COMPONENT_NAME} SHARED ${_${COMPONENT_NAME}_sources} ${_cuda_sources} NO_EXTRAS)
# Alias into the HOOMD namespace so that plugins and symlinked components both work.
add_library(HOOMD::_${COMPONENT_NAME} ALIAS _${COMPONENT_NAME})

if (APPLE)
set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path/../md;@loader_path/../../../..;@loader_path")
else()
set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN/../md;\$ORIGIN/../../../..;\$ORIGIN")
endif()

# Link the library to its dependencies. Add or remove HOOMD extension modules (and/or external C++
# libraries) as needed.
target_link_libraries(_${COMPONENT_NAME} PUBLIC HOOMD::_hoomd)
target_link_libraries(_${COMPONENT_NAME} PUBLIC HOOMD::_md)

# Include HOOMD's internal HIP headers (needed for CPU/GPU builds alike)
target_include_directories(_${COMPONENT_NAME} PRIVATE 
    "${HOOMD_INSTALL_PREFIX}/${PYTHON_SITE_INSTALL_DIR}/include/hoomd/extern/HIP/include"
)

# If HIP is disabled (CPU-only build), we need to define the fallback platform for hip headers
if (NOT ENABLE_HIP)
    target_compile_definitions(_${COMPONENT_NAME} PRIVATE __HIP_PLATFORM_HCC__)
endif()

# Install the library.
install(TARGETS _${COMPONENT_NAME}
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
        )

################ Python only modules
# Copy python modules to the build directory to make it a working python package.
set(files
    src/dybond/__init__.py
    src/dybond/dybond.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
       )

# Copy python files to flat build directory
set(copy_depends)
foreach(file __init__.py dybond.py)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${file}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/dybond/${file}
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/src/dybond/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
        COMMENT "Copying src/dybond/${file} to build directory"
    )
    list(APPEND copy_depends ${CMAKE_CURRENT_BINARY_DIR}/${file})
endforeach()

add_custom_target(copy_${COMPONENT_NAME} ALL DEPENDS ${copy_depends})

# Python tests.
#add_subdirectory(pytest)
