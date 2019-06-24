#version 330 

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 color;

uniform mat4 projMat;
uniform mat4 viewMat;

out vec4 a_color;

void main() {
	gl_Position = projMat * viewMat * vec4(a_position, 1.0);
	a_color = vec4(color / 255.0, 1.0);
}