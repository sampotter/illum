#version 330

in float v_color;
out vec4 f_color;

void main() {
	f_color = vec4(v_color, v_color, v_color, 1);
}
