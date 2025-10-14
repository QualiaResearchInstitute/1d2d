import { spawn } from 'node:child_process';
export class CommandError extends Error {
  command;
  args;
  exitCode;
  stderr;
  constructor(command, args, exitCode, stderr) {
    super(`Command "${command} ${args.join(' ')}" failed with exit code ${exitCode}`);
    this.command = command;
    this.args = [...args];
    this.exitCode = exitCode;
    this.stderr = stderr.toString('utf8');
  }
}
export const runCommand = async (command, args, options = {}) => {
  const child = spawn(command, args, {
    cwd: options.cwd,
    stdio: options.input ? ['pipe', 'pipe', 'pipe'] : ['ignore', 'pipe', 'pipe'],
  });
  const stdoutChunks = [];
  const stderrChunks = [];
  child.stdout?.on('data', (chunk) => stdoutChunks.push(chunk));
  child.stderr?.on('data', (chunk) => stderrChunks.push(chunk));
  if (options.input) {
    const buffer = Buffer.isBuffer(options.input)
      ? options.input
      : Buffer.from(options.input.buffer, options.input.byteOffset ?? 0, options.input.byteLength);
    child.stdin?.write(buffer);
    child.stdin?.end();
  }
  const exitCode = await new Promise((resolve, reject) => {
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
