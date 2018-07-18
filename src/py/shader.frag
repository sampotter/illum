#version 330

in float v_color;
out vec4 f_color;

vec4 cm_hot(float x) {
    float r = clamp(8.0 / 3.0 * x, 0.0, 1.0);
    float g = clamp(8.0 / 3.0 * x - 1.0, 0.0, 1.0);
    float b = clamp(4.0 * x - 3.0, 0.0, 1.0);
    return vec4(r, g, b, 1.0);
}

float cm_jet_red(float x) {
    if (x < 0.7) {
        return 4.0 * x - 1.5;
    } else {
        return -4.0 * x + 4.5;
    }
}

float cm_jet_green(float x) {
    if (x < 0.5) {
        return 4.0 * x - 0.5;
    } else {
        return -4.0 * x + 3.5;
    }
}

float cm_jet_blue(float x) {
    if (x < 0.3) {
       return 4.0 * x + 0.5;
    } else {
       return -4.0 * x + 2.5;
    }
}

void main() {
	// f_color = colormap(v_color);
	f_color = vec4(
		clamp(cm_jet_red(v_color), 0.0, 1.0),
		clamp(cm_jet_green(v_color), 0.0, 1.0),
		clamp(cm_jet_blue(v_color), 0.0, 1.0),
		1.0);
}
