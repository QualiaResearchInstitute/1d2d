import { spawn } from 'node:child_process';

export type RunCommandResult = {
  readonly stdout: Buffer;
  readonly stderr: Buffer;
  readonly code: number;
};

export class CommandError extends Error {
  readonly command: string;
  readonly args: readonly string[];
  readonly exitCode: number;
  readonly stderr: string;

  constructor(command: string, args: readonly string[], exitCode: number, stderr: Buffer) {
    super(`Command "${command} ${args.join(' ')}" failed with exit code ${exitCode}`);
    this.command = command;
    this.args = [...args];
    this.exitCode = exitCode;
    this.stderr = stderr.toString('utf8');
  }
}

export const runCommand = async (
  command: string,
  args: readonly string[],
  options: { input?: ArrayBufferView | Buffer; cwd?: string } = {},
): Promise<RunCommandResult> => {
  const child = spawn(command, args, {
    cwd: options.cwd,
    stdio: options.input ? ['pipe', 'pipe', 'pipe'] : ['ignore', 'pipe', 'pipe'],
  });

  const stdoutChunks: Buffer[] = [];
  const stderrChunks: Buffer[] = [];

  child.stdout?.on('data', (chunk: Buffer) => stdoutChunks.push(chunk));
  child.stderr?.on('data', (chunk: Buffer) => stderrChunks.push(chunk));

  if (options.input) {
    const buffer = Buffer.isBuffer(options.input)
      ? options.input
      : Buffer.from(options.input.buffer, options.input.byteOffset ?? 0, options.input.byteLength);
    child.stdin?.write(buffer);
    child.stdin?.end();
  }

  const exitCode: number = await new Promise((resolve, reject) => {
    child.once('error', reject);
    child.once('close', (code) => resolve(code ?? -1));
  });

  const stdout = Buffer.concat(stdoutChunks);
  const stderr = Buffer.concat(stderrChunks);

  if (exitCode !== 0) {
    throw new CommandError(command, [...args], exitCode, stderr);
  }

  return { stdout, stderr, code: exitCode };
};
