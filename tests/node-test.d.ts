declare module "node:test" {
  const test: any;
  export default test;
}

declare module "node:assert/strict" {
  const assert: any;
  export default assert;
}

declare module "node:fs/promises" {
  export const readFile: any;
}

declare module "node:path" {
  export const join: (...args: any[]) => string;
}

declare const process: {
  cwd(): string;
};
