#version 330 

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

uniform mat4 projMat;
uniform mat4 viewMat;

out vec4 a_color;

void main() {
	gl_Position = projMat * viewMat * vec4(a_position, 1.0);
	vec3 normal = vec3(a_normal.x, a_normal.y, a_normal.z); 
	a_color = vec4((normal + 1) / 2, 1);
}