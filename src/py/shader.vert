#version 330

uniform mat4 Mvp;

in vec4 in_vert;
out float v_color;

void main() {
	v_color = in_vert.w;
	gl_Position = Mvp * vec4(in_vert.xyz, 1.0);
}
