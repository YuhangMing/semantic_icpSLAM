option(WITH_OPENNI ON)

if(${WITH_OPENNI})
  message("-- USING Camera: Linking OpenNI")
  target_sources(fusion
  PRIVATE
    src/input/oni_camera.cpp
  )

  target_include_directories(fusion
  PUBLIC
    /usr/local/include/OpenNI2
  )

  if(NOT ${Pangolin_FOUND})
    target_link_directories(fusion
    PUBLIC
      /usr/local/lib/OpenNI2
    )

    target_link_libraries(fusion
    PUBLIC
      OpenNI2
    )
  endif()
endif()
