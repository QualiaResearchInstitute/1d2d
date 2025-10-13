struct QcdKernelUniforms {
  latticeSize: vec2<u32>,
  parityAndRelax: vec2<u32>,
  strides: vec4<u32>,
  seedScope: vec4<u32>,
  betaParams: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> uLattice : array<f32>;
@group(0) @binding(1) var uRngSeeds : texture_2d<u32>;
@group(0) @binding(2) var<uniform> uParams : QcdKernelUniforms;

const DIM: u32 = 3u;
const COMPLEX_COMPONENTS: u32 = 2u;
const MATRIX_SIZE: u32 = DIM * DIM;
const FLOATS_PER_MATRIX: u32 = MATRIX_SIZE * COMPLEX_COMPONENTS;
const STAPLE_MULTIPLICITY: f32 = 2.0;
const WORKGROUP_SIZE_X: u32 = 8u;
const WORKGROUP_SIZE_Y: u32 = 8u;
const WORKGROUP_SIZE: u32 = WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y;
const EPSILON: f32 = 1e-8;
const TAU: f32 = 6.283185307179586;

type Complex = vec2<f32>;
type Matrix3x3 = array<Complex, 9u>;

var<workgroup> sharedLinksX : array<Complex, MATRIX_SIZE * WORKGROUP_SIZE>;
var<workgroup> sharedLinksY : array<Complex, MATRIX_SIZE * WORKGROUP_SIZE>;

fn complex(re: f32, im: f32) -> Complex {
  return vec2<f32>(re, im);
}

fn complexAdd(a: Complex, b: Complex) -> Complex {
  return a + b;
}

fn complexSub(a: Complex, b: Complex) -> Complex {
  return a - b;
}

fn complexMul(a: Complex, b: Complex) -> Complex {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn complexConj(value: Complex) -> Complex {
  return vec2<f32>(value.x, -value.y);
}

fn complexScale(value: Complex, scalar: f32) -> Complex {
  return vec2<f32>(value.x * scalar, value.y * scalar);
}

fn complexAbs2(value: Complex) -> f32 {
  return dot(value, value);
}

fn complexDiv(a: Complex, b: Complex) -> Complex {
  let denom = complexAbs2(b);
  if (denom <= EPSILON) {
    return vec2<f32>(0.0, 0.0);
  }
  let num = complexMul(a, complexConj(b));
  let inv = 1.0 / denom;
  return vec2<f32>(num.x * inv, num.y * inv);
}

fn complexMulImag(value: Complex) -> Complex {
  return vec2<f32>(-value.y, value.x);
}

fn complexMulNegImag(value: Complex) -> Complex {
  return vec2<f32>(value.y, -value.x);
}

fn matrixIndex(row: u32, col: u32) -> u32 {
  return row * DIM + col;
}

fn matrixZero() -> Matrix3x3 {
  var result: Matrix3x3;
  for (var idx: u32 = 0u; idx < MATRIX_SIZE; idx = idx + 1u) {
    result[idx] = vec2<f32>(0.0, 0.0);
  }
  return result;
}

fn matrixIdentity() -> Matrix3x3 {
  var result = matrixZero();
  for (var i: u32 = 0u; i < DIM; i = i + 1u) {
    let idx = matrixIndex(i, i);
    result[idx] = vec2<f32>(1.0, 0.0);
  }
  return result;
}

fn matrixClone(matrix: Matrix3x3) -> Matrix3x3 {
  var result: Matrix3x3;
  for (var idx: u32 = 0u; idx < MATRIX_SIZE; idx = idx + 1u) {
    result[idx] = matrix[idx];
  }
  return result;
}

fn matrixAdd(a: Matrix3x3, b: Matrix3x3) -> Matrix3x3 {
  var result: Matrix3x3;
  for (var idx: u32 = 0u; idx < MATRIX_SIZE; idx = idx + 1u) {
    result[idx] = complexAdd(a[idx], b[idx]);
  }
  return result;
}

fn matrixSub(a: Matrix3x3, b: Matrix3x3) -> Matrix3x3 {
  var result: Matrix3x3;
  for (var idx: u32 = 0u; idx < MATRIX_SIZE; idx = idx + 1u) {
    result[idx] = complexSub(a[idx], b[idx]);
  }
  return result;
}

fn matrixScale(matrix: Matrix3x3, scalar: f32) -> Matrix3x3 {
  var result: Matrix3x3;
  for (var idx: u32 = 0u; idx < MATRIX_SIZE; idx = idx + 1u) {
    result[idx] = complexScale(matrix[idx], scalar);
  }
  return result;
}

fn matrixAddWeighted(base: Matrix3x3, addition: Matrix3x3, weight: f32) -> Matrix3x3 {
  var result: Matrix3x3;
  for (var idx: u32 = 0u; idx < MATRIX_SIZE; idx = idx + 1u) {
    result[idx] = complexAdd(base[idx], complexScale(addition[idx], weight));
  }
  return result;
}

fn matrixWeightedSum(a: Matrix3x3, weightA: f32, b: Matrix3x3, weightB: f32) -> Matrix3x3 {
  var result: Matrix3x3;
  for (var idx: u32 = 0u; idx < MATRIX_SIZE; idx = idx + 1u) {
    let termA = complexScale(a[idx], weightA);
    let termB = complexScale(b[idx], weightB);
    result[idx] = complexAdd(termA, termB);
  }
  return result;
}

fn matrixMultiply(a: Matrix3x3, b: Matrix3x3) -> Matrix3x3 {
  var result = matrixZero();
  for (var row: u32 = 0u; row < DIM; row = row + 1u) {
    for (var col: u32 = 0u; col < DIM; col = col + 1u) {
      var sum = vec2<f32>(0.0, 0.0);
      for (var k: u32 = 0u; k < DIM; k = k + 1u) {
        let lhs = a[matrixIndex(row, k)];
        let rhs = b[matrixIndex(k, col)];
        sum = complexAdd(sum, complexMul(lhs, rhs));
      }
      result[matrixIndex(row, col)] = sum;
    }
  }
  return result;
}

fn conjugateTranspose(matrix: Matrix3x3) -> Matrix3x3 {
  var result: Matrix3x3;
  for (var row: u32 = 0u; row < DIM; row = row + 1u) {
    for (var col: u32 = 0u; col < DIM; col = col + 1u) {
      let idx = matrixIndex(row, col);
      let src = matrixIndex(col, row);
      result[idx] = complexConj(matrix[src]);
    }
  }
  return result;
}

fn matrixMinusIdentity(matrix: Matrix3x3) -> Matrix3x3 {
  var result: Matrix3x3;
  for (var row: u32 = 0u; row < DIM; row = row + 1u) {
    for (var col: u32 = 0u; col < DIM; col = col + 1u) {
      let idx = matrixIndex(row, col);
      let value = matrix[idx];
      if (row == col) {
        result[idx] = vec2<f32>(value.x - 1.0, value.y);
      } else {
        result[idx] = value;
      }
    }
  }
  return result;
}

fn matrixOneNorm(matrix: Matrix3x3) -> f32 {
  var maxVal = 0.0;
  for (var col: u32 = 0u; col < DIM; col = col + 1u) {
    var sum = 0.0;
    for (var row: u32 = 0u; row < DIM; row = row + 1u) {
      let entry = matrix[matrixIndex(row, col)];
      sum = sum + length(entry);
    }
    maxVal = max(maxVal, sum);
  }
  return maxVal;
}

fn frobeniusNorm(matrix: Matrix3x3) -> f32 {
  var sum = 0.0;
  for (var idx: u32 = 0u; idx < MATRIX_SIZE; idx = idx + 1u) {
    let entry = matrix[idx];
    sum = sum + dot(entry, entry);
  }
  return sqrt(sum);
}

fn determinant(matrix: Matrix3x3) -> Complex {
  let a = matrix[matrixIndex(0u, 0u)];
  let b = matrix[matrixIndex(0u, 1u)];
  let cEntry = matrix[matrixIndex(0u, 2u)];
  let d = matrix[matrixIndex(1u, 0u)];
  let e = matrix[matrixIndex(1u, 1u)];
  let f = matrix[matrixIndex(1u, 2u)];
  let g = matrix[matrixIndex(2u, 0u)];
  let h = matrix[matrixIndex(2u, 1u)];
  let i = matrix[matrixIndex(2u, 2u)];
  let term1 = complexMul(a, complexSub(complexMul(e, i), complexMul(f, h)));
  let term2 = complexMul(b, complexSub(complexMul(d, i), complexMul(f, g)));
  let term3 = complexMul(cEntry, complexSub(complexMul(d, h), complexMul(e, g)));
  return complexAdd(complexSub(term1, term2), term3);
}

fn normalizeDeterminant(matrix: Matrix3x3) -> Matrix3x3 {
  let det = determinant(matrix);
  var angle = -atan2(det.y, det.x) / f32(DIM);
  if (isNan(angle) || isInf(angle)) {
    angle = 0.0;
  }
  let correction = vec2<f32>(cos(angle), sin(angle));
  var result: Matrix3x3;
  for (var idx: u32 = 0u; idx < MATRIX_SIZE; idx = idx + 1u) {
    result[idx] = complexMul(matrix[idx], correction);
  }
  return result;
}

fn polarInverseSqrt(matrix: Matrix3x3) -> Matrix3x3 {
  var trace = 0.0;
  for (var i: u32 = 0u; i < DIM; i = i + 1u) {
    trace = trace + matrix[matrixIndex(i, i)].x;
  }
  var scale = trace / f32(DIM);
  if (!(scale > 0.0) || isNan(scale) || isInf(scale)) {
    scale = 1.0;
  }
  let invScale = 1.0 / scale;
  var Y = matrixScale(matrix, invScale);
  var Z = matrixIdentity();
  var residual = 1e9;

  for (var iter: u32 = 0u; iter < 12u; iter = iter + 1u) {
    let YZ = matrixMultiply(Z, Y);
    var correction = matrixZero();
    for (var row: u32 = 0u; row < DIM; row = row + 1u) {
      for (var col: u32 = 0u; col < DIM; col = col + 1u) {
        let idx = matrixIndex(row, col);
        correction[idx] = vec2<f32>(-YZ[idx].x, -YZ[idx].y);
      }
      let diag = matrixIndex(row, row);
      correction[diag].x = correction[diag].x + f32(DIM);
    }
    let halfCorrection = matrixScale(correction, 0.5);
    let nextY = matrixMultiply(Y, halfCorrection);
    let nextZ = matrixMultiply(halfCorrection, Z);
    let gram = matrixMultiply(conjugateTranspose(nextY), nextY);
    let errorMatrix = matrixMinusIdentity(gram);
    let error = frobeniusNorm(errorMatrix);
    Y = nextY;
    Z = nextZ;
    if (error < 1e-6 || abs(error - residual) < 1e-8) {
      break;
    }
    residual = error;
  }

  let scaleFactor = 1.0 / sqrt(scale);
  return matrixScale(Z, scaleFactor);
}

fn polarReunitarize(matrix: Matrix3x3) -> Matrix3x3 {
  let base = matrixClone(matrix);
  let gram = matrixMultiply(conjugateTranspose(base), base);
  let invSqrt = polarInverseSqrt(gram);
  let unitary = matrixMultiply(base, invSqrt);
  return normalizeDeterminant(unitary);
}

fn su3Project(matrix: Matrix3x3) -> Matrix3x3 {
  return polarReunitarize(matrix);
}

fn su3Mul(a: Matrix3x3, b: Matrix3x3) -> Matrix3x3 {
  return matrixMultiply(a, b);
}

fn su3FrobeniusNorm(matrix: Matrix3x3) -> f32 {
  return frobeniusNorm(matrix);
}

fn wrapInc(value: u32, limit: u32) -> u32 {
  let next = value + 1u;
  if (next < limit) {
    return next;
  }
  return 0u;
}

fn wrapDec(value: u32, limit: u32) -> u32 {
  if (value == 0u) {
    return limit - 1u;
  }
  return value - 1u;
}

fn workgroupTileOrigin() -> vec2<u32> {
  let originX = workgroup_id.x * WORKGROUP_SIZE_X;
  let originY = workgroup_id.y * WORKGROUP_SIZE_Y;
  return vec2<u32>(originX, originY);
}

fn workgroupTileSize(width: u32, height: u32, origin: vec2<u32>) -> vec2<u32> {
  let remainingX = width - min(width, origin.x);
  let remainingY = height - min(height, origin.y);
  let sizeX = min(WORKGROUP_SIZE_X, remainingX);
  let sizeY = min(WORKGROUP_SIZE_Y, remainingY);
  return vec2<u32>(sizeX, sizeY);
}

fn latticeSiteStride() -> u32 {
  return uParams.strides.x;
}

fn latticeLinkStride() -> u32 {
  return uParams.strides.y;
}

fn latticeRowStride() -> u32 {
  return uParams.strides.z;
}

fn latticeComplexStride() -> u32 {
  return uParams.strides.w;
}

fn siteBaseIndex(x: u32, y: u32) -> u32 {
  let width = uParams.latticeSize.x;
  return (y * width + x) * latticeSiteStride();
}

fn axisOffset(axis: u32) -> u32 {
  return select(latticeLinkStride(), 0u, axis == 0u);
}

fn loadMatrixFromBuffer(x: u32, y: u32, axis: u32) -> Matrix3x3 {
  let base = siteBaseIndex(x, y) + axisOffset(axis);
  var matrix: Matrix3x3;
  var cursor = base;
  for (var row: u32 = 0u; row < DIM; row = row + 1u) {
    for (var col: u32 = 0u; col < DIM; col = col + 1u) {
      let idx = matrixIndex(row, col);
      let re = uLattice[cursor];
      let im = uLattice[cursor + 1u];
      matrix[idx] = vec2<f32>(re, im);
      cursor = cursor + latticeComplexStride();
    }
    cursor = base + (row + 1u) * latticeRowStride();
  }
  return matrix;
}

fn storeMatrixToBuffer(x: u32, y: u32, axis: u32, matrix: Matrix3x3) {
  let base = siteBaseIndex(x, y) + axisOffset(axis);
  var cursor = base;
  for (var row: u32 = 0u; row < DIM; row = row + 1u) {
    for (var col: u32 = 0u; col < DIM; col = col + 1u) {
      let idx = matrixIndex(row, col);
      let value = matrix[idx];
      uLattice[cursor] = value.x;
      uLattice[cursor + 1u] = value.y;
      cursor = cursor + latticeComplexStride();
    }
    cursor = base + (row + 1u) * latticeRowStride();
  }
}

fn sharedBaseIndex(localIndex: u32) -> u32 {
  return localIndex * MATRIX_SIZE;
}

fn loadFromShared(coord: vec2<u32>, axis: u32, origin: vec2<u32>, size: vec2<u32>) -> Matrix3x3 {
  let withinX = coord.x >= origin.x && coord.x < origin.x + size.x;
  let withinY = coord.y >= origin.y && coord.y < origin.y + size.y;
  if (withinX && withinY) {
    let tileWidth = size.x;
    let localX = coord.x - origin.x;
    let localY = coord.y - origin.y;
    let localIndex = localY * tileWidth + localX;
    let base = sharedBaseIndex(localIndex);
    var matrix: Matrix3x3;
    for (var idx: u32 = 0u; idx < MATRIX_SIZE; idx = idx + 1u) {
      if (axis == 0u) {
        matrix[idx] = sharedLinksX[base + idx];
      } else {
        matrix[idx] = sharedLinksY[base + idx];
      }
    }
    return matrix;
  }
  return loadMatrixFromBuffer(coord.x, coord.y, axis);
}

fn computeStaple(
  coord: vec2<u32>,
  axis: u32,
  origin: vec2<u32>,
  size: vec2<u32>
) -> Matrix3x3 {
  let width = uParams.latticeSize.x;
  let height = uParams.latticeSize.y;
  var staple = matrixZero();

  if (axis == 0u) {
    let xp1 = wrapInc(coord.x, width);
    let yp1 = wrapInc(coord.y, height);
    let ym1 = wrapDec(coord.y, height);

    let Uy = loadFromShared(coord, 1u, origin, size);
    let UxForward = loadFromShared(vec2<u32>(coord.x, yp1), 0u, origin, size);
    let UyForward = loadFromShared(vec2<u32>(xp1, coord.y), 1u, origin, size);
    let forward = su3Mul(su3Mul(Uy, UxForward), conjugateTranspose(UyForward));
    staple = matrixAdd(staple, forward);

    let UyBackward = conjugateTranspose(loadFromShared(vec2<u32>(coord.x, ym1), 1u, origin, size));
    let UxBackward = loadFromShared(vec2<u32>(coord.x, ym1), 0u, origin, size);
    let UyBackwardNeighbor = loadFromShared(vec2<u32>(xp1, ym1), 1u, origin, size);
    let backward = su3Mul(su3Mul(UyBackward, UxBackward), UyBackwardNeighbor);
    staple = matrixAdd(staple, backward);
  } else {
    let yp1 = wrapInc(coord.y, height);
    let ym1 = wrapDec(coord.y, height);
    let xp1 = wrapInc(coord.x, width);
    let xm1 = wrapDec(coord.x, width);

    let Ux = loadFromShared(coord, 0u, origin, size);
    let UyForward = loadFromShared(vec2<u32>(xp1, coord.y), 1u, origin, size);
    let UxForward = loadFromShared(vec2<u32>(coord.x, yp1), 0u, origin, size);
    let forward = su3Mul(su3Mul(Ux, UyForward), conjugateTranspose(UxForward));
    staple = matrixAdd(staple, forward);

    let UxBackward = conjugateTranspose(loadFromShared(vec2<u32>(xm1, coord.y), 0u, origin, size));
    let UyBackward = loadFromShared(vec2<u32>(xm1, coord.y), 1u, origin, size);
    let UxBackwardNeighbor = loadFromShared(vec2<u32>(xm1, yp1), 0u, origin, size);
    let backward = su3Mul(su3Mul(UxBackward, UyBackward), UxBackwardNeighbor);
    staple = matrixAdd(staple, backward);
  }

  return staple;
}

fn computeNoiseMix(beta: f32, stapleNorm: f32) -> vec3<f32> {
  let scaled = max(0.0, beta) * stapleNorm;
  let align = tanh(scaled / 12.0);
  let stapleWeight = align;
  let linkWeight = max(0.0, 1.0 - stapleWeight);
  let noiseWeight = sqrt(max(0.0, (1.0 - align) * 0.25));
  return vec3<f32>(stapleWeight, linkWeight, noiseWeight);
}

fn mulHi(a: u32, b: u32) -> u32 {
  let ah = a >> 16u;
  let al = a & 0xffffu;
  let bh = b >> 16u;
  let bl = b & 0xffffu;

  let ahbh = ah * bh;
  let ahbl = ah * bl;
  let albh = al * bh;
  let albl = al * bl;

  let carry = ((albl >> 16u) + (albh & 0xffffu) + (ahbl & 0xffffu)) >> 16u;
  return ahbh + (albh >> 16u) + (ahbl >> 16u) + carry;
}

fn philoxRound(counter: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
  let mul0: u32 = 0xD2511F53u;
  let mul1: u32 = 0xCD9E8D57u;

  let hi0 = mulHi(mul0, counter.x);
  let lo0 = mul0 * counter.x;
  let hi1 = mulHi(mul1, counter.z);
  let lo1 = mul1 * counter.z;

  return vec4<u32>(
    hi1 ^ counter.y ^ key.x,
    lo1,
    hi0 ^ counter.w ^ key.y,
    lo0
  );
}

fn philox10(counter: vec4<u32>, keyInit: vec2<u32>) -> vec4<u32> {
  var ctr = counter;
  var key = keyInit;
  let weyl: u32 = 0x9E3779B9u;
  for (var round: u32 = 0u; round < 10u; round = round + 1u) {
    ctr = philoxRound(ctr, key);
    key.x = key.x + weyl;
    key.y = key.y + weyl;
  }
  return ctr;
}

struct PhiloxState {
  counter: vec4<u32>,
  key: vec2<u32>,
  lane: u32,
  block: vec4<u32>,
};

fn initPhilox(coord: vec2<u32>, axis: u32) -> PhiloxState {
  let seedPixel = textureLoad(uRngSeeds, vec2<i32>(i32(coord.x), i32(coord.y)), 0);
  let width = uParams.latticeSize.x;
  let siteIndex = coord.y * width + coord.x;
  let seedKey = vec2<u32>(seedPixel.x ^ uParams.seedScope.x, seedPixel.y ^ uParams.seedScope.y);
  let sweep = uParams.seedScope.z;
  let salt = seedPixel.w;
  let counter = vec4<u32>(
    siteIndex ^ seedPixel.z,
    sweep,
    axis ^ salt,
    0u
  );
  var state = PhiloxState(counter, seedKey, 4u, vec4<u32>(0u));
  return state;
}

fn selectLane(vec: vec4<u32>, lane: u32) -> u32 {
  switch lane {
    case 0u: {
      return vec.x;
    }
    case 1u: {
      return vec.y;
    }
    case 2u: {
      return vec.z;
    }
    default: {
      return vec.w;
    }
  }
}

fn advancePhilox(state: ptr<function, PhiloxState>) {
  (*state).block = philox10((*state).counter, (*state).key);
  (*state).counter.w = (*state).counter.w + 1u;
  (*state).lane = 0u;
}

fn philoxNext(state: ptr<function, PhiloxState>) -> f32 {
  if ((*state).lane >= 4u) {
    advancePhilox(state);
  }
  let lane = (*state).lane;
  let value = selectLane((*state).block, lane);
  (*state).lane = lane + 1u;
  return (f32(value) + 0.5) * 2.3283064365386963e-10;
}

fn sampleGaussianPair(state: ptr<function, PhiloxState>) -> vec2<f32> {
  var u1 = philoxNext(state);
  var u2 = philoxNext(state);
  u1 = max(u1, 1e-7);
  let radius = sqrt(-2.0 * log(u1));
  let theta = TAU * u2;
  return vec2<f32>(radius * cos(theta), radius * sin(theta));
}

fn su3Haar(state: ptr<function, PhiloxState>, variance: f32) -> Matrix3x3 {
  var result: Matrix3x3;
  let scale = sqrt(max(variance, 0.0));
  for (var idx: u32 = 0u; idx < MATRIX_SIZE; idx = idx + 1u) {
    let gaussian = sampleGaussianPair(state);
    result[idx] = vec2<f32>(gaussian.x * scale, gaussian.y * scale);
  }
  return su3Project(result);
}

fn applyHeatbath(
  coord: vec2<u32>,
  axis: u32,
  origin: vec2<u32>,
  size: vec2<u32>,
  stateSeed: ptr<function, PhiloxState>
) {
  let current = loadFromShared(coord, axis, origin, size);
  let staple = computeStaple(coord, axis, origin, size);
  let stapleProjected = su3Project(staple);
  let stapleNorm = su3FrobeniusNorm(staple) / STAPLE_MULTIPLICITY;
  let weights = computeNoiseMix(uParams.betaParams.x, stapleNorm);

  var proposal = matrixWeightedSum(current, weights.y, stapleProjected, weights.x);

  if (weights.z > 1e-6) {
    let noise = su3Haar(stateSeed, 1.0);
    proposal = matrixAddWeighted(proposal, noise, weights.z);
  }

  var updated = su3Project(proposal);
  let reflections = uParams.parityAndRelax.y;

  if (reflections > 0u) {
    for (var step: u32 = 0u; step < reflections; step = step + 1u) {
      let reflected = matrixAddWeighted(matrixScale(stapleProjected, 2.0), updated, -1.0);
      updated = su3Project(reflected);
    }
  }

  storeMatrixToBuffer(coord.x, coord.y, axis, updated);
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
  let width = uParams.latticeSize.x;
  let height = uParams.latticeSize.y;
  let x = global_id.x;
  let y = global_id.y;

  if (x >= width || y >= height) {
    return;
  }

  let parity = (x + y) & 1u;
  if (parity != uParams.parityAndRelax.x) {
    return;
  }

  let origin = workgroupTileOrigin();
  let size = workgroupTileSize(width, height, origin);
  let localLinear = local_id.y * WORKGROUP_SIZE_X + local_id.x;
  let baseIndex = sharedBaseIndex(localLinear);

  let coord = vec2<u32>(x, y);
  let linkX = loadMatrixFromBuffer(x, y, 0u);
  let linkY = loadMatrixFromBuffer(x, y, 1u);

  for (var idx: u32 = 0u; idx < MATRIX_SIZE; idx = idx + 1u) {
    sharedLinksX[baseIndex + idx] = linkX[idx];
    sharedLinksY[baseIndex + idx] = linkY[idx];
  }

  workgroupBarrier();

  let axisSelect = min(uParams.seedScope.w, 1u);
  var philox = initPhilox(coord, axisSelect);
  philox.lane = 4u;
  applyHeatbath(coord, axisSelect, origin, size, &philox);
}
