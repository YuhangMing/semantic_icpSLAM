#version 330 

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

uniform mat4 projMat;
uniform mat4 viewMat;

out vec4 a_color;

void main() {
	vec3 lightpos = vec3(0.0, 0.0, -1.0);
	gl_Position = projMat * viewMat * vec4(a_position, 1.0);
	vec3 pos = a_position;
  	vec3 normal = a_normal;
	const float ka = 0.3;
	const float kd = 0.5;
	const float ks = 0.2;
	const float n = 20.0;
	const float ax = 1.0;
	const float dx = 1.0;
	const float sx = 1.0;
	const float lx = 1.0;
	vec3 L = normalize(lightpos - pos);
	vec3 V = normalize(vec3(0.0) - pos);
	vec3 R = normalize(2 * normal * dot(normal, L) - L);
	float i1 = ax * ka * dx;
	float i2 = lx * kd * dx * max(0.0, dot(normal, L));
	float i3 = lx * ks * sx * pow(max(0.0, dot(R, V)), n);
	float Ix = max(0.0, min(255.0, i1 + i2 + i3));
	a_color = vec4(Ix, Ix, Ix, 1.0);
} 