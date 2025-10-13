declare module 'node:test' {
  const test: any;
  export default test;
}

declare module 'node:assert/strict' {
  const assert: any;
  export default assert;
}

declare module 'node:fs/promises' {
  export const readFile: any;
}

declare module 'node:path' {
  export const join: (...args: any[]) => string;
}

declare module 'jsdom' {
  export class JSDOM {
    constructor(html?: string, options?: any);
    window: Window & typeof globalThis;
  }
}

declare module 'axe-core' {
  const axe: {
    run: (
      context?: Element | Document,
      options?: unknown,
    ) => Promise<{
      violations: Array<{ id: string; nodes: Array<unknown> }>;
    }>;
  };
  export default axe;
}

declare const process: {
  cwd(): string;
};
