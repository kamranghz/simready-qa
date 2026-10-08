cmake_minimum_required(VERSION 3.16)
project(SimReadyQA LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to an extracted MuJoCo release (contains include/, lib/, bin/)
# or to a CMake install prefix that provides mujocoConfig.cmake.
set(MUJOCO_DIR "" CACHE PATH "MuJoCo root directory")

find_package(mujoco CONFIG QUIET)

if(NOT mujoco_FOUND)
  if(MUJOCO_DIR STREQUAL "")
    message(FATAL_ERROR
      "MuJoCo was not found.\n"
      "Set MUJOCO_DIR to the extracted MuJoCo folder, for example:\n"
      "  cmake -S . -B build -DMUJOCO_DIR=C:/path/to/mujoco")
  endif()

  find_path(MUJOCO_INCLUDE_DIR
    NAMES mujoco/mujoco.h
    PATHS "${MUJOCO_DIR}/include"
    NO_DEFAULT_PATH)

  find_library(MUJOCO_LIBRARY
    NAMES mujoco
    PATHS "${MUJOCO_DIR}/lib" "${MUJOCO_DIR}/bin"
    NO_DEFAULT_PATH)

  if(NOT MUJOCO_INCLUDE_DIR OR NOT MUJOCO_LIBRARY)
    message(FATAL_ERROR
      "Could not find MuJoCo headers or library under ${MUJOCO_DIR}.\n"
      "Expected include/mujoco/mujoco.h and lib/mujoco.lib (Windows) "
      "or lib/libmujoco.so (Linux).")
  endif()

  add_library(mujoco::mujoco SHARED IMPORTED)
  set_target_properties(mujoco::mujoco PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}")

  if(WIN32)
    set_target_properties(mujoco::mujoco PROPERTIES
      IMPORTED_IMPLIB "${MUJOCO_LIBRARY}"
      IMPORTED_LOCATION "${MUJOCO_DIR}/bin/mujoco.dll")
  else()
    set_target_properties(mujoco::mujoco PROPERTIES
      IMPORTED_LOCATION "${MUJOCO_LIBRARY}")
  endif()
endif()

add_executable(simready_qa
  src/main.cpp
  src/environment.cpp)
target_include_directories(simready_qa PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(simready_qa PRIVATE mujoco::mujoco)

# Bake in source-tree paths so the program does not depend on cwd.
target_compile_definitions(simready_qa PRIVATE
  "DEFAULT_MODEL_PATH=\"${CMAKE_SOURCE_DIR}/assets/humanoid_12dof.xml\""
  "ASSETS_DIR=\"${CMAKE_SOURCE_DIR}/assets\""
  "OUTPUTS_DIR=\"${CMAKE_SOURCE_DIR}/outputs\"")

# Windows loads mujoco.dll from the same folder as the .exe.
if(WIN32)
  add_custom_command(TARGET simready_qa POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE:mujoco::mujoco>"
      "$<TARGET_FILE_DIR:simready_qa>")
endif()
