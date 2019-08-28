project(openni_example)

add_executable(openni main.cpp)

target_link_libraries(openni
PRIVATE
    fusion
)