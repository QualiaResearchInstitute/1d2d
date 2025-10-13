struct Complex {
  re: f32,
  im: f32,
};

struct KernelUniforms {
  vectorCount: u32,
  stride: u32,
  _padding: vec2<u32>,
};

@group(0) @binding(0) var<storage, read> uUnitary : array<Complex>;
@group(0) @binding(1) var<storage, read> uInput : array<Complex>;
@group(0) @binding(2) var<storage, read_write> uOutput : array<Complex>;
@group(0) @binding(3) var<uniform> uParams : KernelUniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let index = global_id.x;
  if (index >= uParams.vectorCount) {
    return;
  }

  let stride = uParams.stride;
  let base = index * stride;

  for (var row: u32 = 0u; row < 7u; row = row + 1u) {
    var sum = Complex(0.0, 0.0);
    for (var col: u32 = 0u; col < 7u; col = col + 1u) {
      let matIndex = row * 7u + col;
      let vecIndex = base + col;
      let coeff = uUnitary[matIndex];
      let value = uInput[vecIndex];
      let mulRe = coeff.re * value.re - coeff.im * value.im;
      let mulIm = coeff.re * value.im + coeff.im * value.re;
      sum.re = sum.re + mulRe;
      sum.im = sum.im + mulIm;
    }
    uOutput[base + row] = sum;
  }
}
