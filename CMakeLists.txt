project(${SKBUILD_PROJECT_NAME} LANGUAGES C)

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)

add_custom_command(
  OUTPUT pycblas.c
  COMMENT
  "Making ${CMAKE_CURRENT_BINARY_DIR}/src/pycblas.c from ${CMAKE_CURRENT_SOURCE_DIR}/src/pycblas.pyx"
  COMMAND Python::Interpreter -m cython
          "${CMAKE_CURRENT_SOURCE_DIR}/src/pycblas.pyx" --output-file pycblas.c
  DEPENDS pycblas.pyx
  VERBATIM)

python_add_library(pycblas MODULE pycblas.c WITH_SOABI)
include_directories(src)

find_library(CBLAS_LIBRARY NAMES cblas HINTS Python_SITELIB REQUIRED)
include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(cblas DEFAULT_MSG CBLAS_LIBRARY)
mark_as_advanced(CBLAS_LIBRARY)
add_library(cblas::cblas SHARED IMPORTED)

set_target_properties(cblas::cblas PROPERTIES IMPORTED_LOCATION ${CBLAS_LIBRARY})

target_link_libraries(pycblas PRIVATE cblas::cblas)


install(TARGETS pycblas DESTINATION .)

