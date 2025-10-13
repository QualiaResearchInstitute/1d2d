#version 310 es
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct Complex {
  float re;
  float im;
};

layout(std430, binding = 0) readonly buffer UnitaryBuffer {
  Complex data[];
} uUnitary;

layout(std430, binding = 1) readonly buffer InputBuffer {
  Complex data[];
} uInput;

layout(std430, binding = 2) writeonly buffer OutputBuffer {
  Complex data[];
} uOutput;

layout(std140, binding = 3) uniform KernelUniforms {
  uint vectorCount;
  uint stride;
};

void main() {
  uint index = gl_GlobalInvocationID.x;
  if (index >= vectorCount) {
    return;
  }

  uint base = index * stride;
  for (uint row = 0u; row < 7u; ++row) {
    float sumRe = 0.0;
    float sumIm = 0.0;
    for (uint col = 0u; col < 7u; ++col) {
      uint matIndex = row * 7u + col;
      uint vecIndex = base + col;
      Complex coeff = uUnitary.data[matIndex];
      Complex value = uInput.data[vecIndex];
      float mulRe = coeff.re * value.re - coeff.im * value.im;
      float mulIm = coeff.re * value.im + coeff.im * value.re;
      sumRe += mulRe;
      sumIm += mulIm;
    }
    uOutput.data[base + row].re = sumRe;
    uOutput.data[base + row].im = sumIm;
  }
}
