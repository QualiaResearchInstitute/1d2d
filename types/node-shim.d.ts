declare module 'node:fs' {
  export function mkdirSync(path: string, options?: { recursive?: boolean }): void;
  export function writeFileSync(path: string, data: string | NodeJS.ArrayBufferView): void;
  export function readFileSync(path: string, encoding: 'utf8'): string;
}

declare module 'node:path' {
  export function join(...parts: string[]): string;
  export function dirname(path: string): string;
}

declare module 'node:url' {
  export function fileURLToPath(url: string | URL): string;
}

declare namespace NodeJS {
  interface ArrayBufferView {
    readonly buffer: ArrayBufferLike;
  }
}

declare const Buffer: {
  from(input: string, encoding?: string): NodeJS.ArrayBufferView;
  alloc(size: number): NodeJS.ArrayBufferView;
  concat(chunks: NodeJS.ArrayBufferView[]): NodeJS.ArrayBufferView;
};

declare const process: {
  exitCode: number | undefined;
};
