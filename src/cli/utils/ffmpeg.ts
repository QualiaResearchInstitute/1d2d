import { basename } from 'node:path';

import { runCommand } from './exec.js';

export type MediaProbe = {
  readonly width: number;
  readonly height: number;
  readonly fps?: number;
  readonly frameCount?: number;
};

const utf8 = new TextDecoder();

export const probeMedia = async (ffprobe: string, input: string): Promise<MediaProbe> => {
  const args = [
    '-v',
    'error',
    '-select_streams',
    'v:0',
    '-show_entries',
    'stream=width,height,r_frame_rate,nb_frames',
    '-of',
    'json',
    input,
  ];
  const { stdout } = await runCommand(ffprobe, args);
  const payload = JSON.parse(utf8.decode(stdout)) as {
    streams?: Array<{
      width?: number;
      height?: number;
      r_frame_rate?: string;
      nb_frames?: string;
    }>;
  };
  const stream = payload.streams?.[0];
  if (!stream || typeof stream.width !== 'number' || typeof stream.height !== 'number') {
    throw new Error(`ffprobe failed to derive dimensions for ${basename(input)}`);
  }
  let fps: number | undefined;
  if (stream.r_frame_rate && stream.r_frame_rate.includes('/')) {
    const [num, den] = stream.r_frame_rate.split('/', 2).map((part) => Number(part));
    if (Number.isFinite(num) && Number.isFinite(den) && den !== 0) {
      fps = num / den;
    }
  }
  let frameCount: number | undefined;
  if (stream.nb_frames && stream.nb_frames.length > 0) {
    const parsed = Number(stream.nb_frames);
    if (Number.isFinite(parsed)) {
      frameCount = parsed;
    }
  }
  return {
    width: stream.width,
    height: stream.height,
    fps,
    frameCount,
  };
};

export const decodeFrame = async (
  ffmpeg: string,
  input: string,
  width: number,
  height: number,
  options: { timeSeconds?: number } = {},
): Promise<Uint8Array> => {
  const args: string[] = ['-v', 'error'];
  if (
    options.timeSeconds != null &&
    Number.isFinite(options.timeSeconds) &&
    options.timeSeconds >= 0
  ) {
    args.push('-ss', options.timeSeconds.toString());
  }
  args.push('-i', input, '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-');
  const { stdout } = await runCommand(ffmpeg, args);
  if (stdout.byteLength !== width * height * 4) {
    throw new Error(
      `Expected ${width * height * 4} bytes for decoded frame, received ${stdout.byteLength}`,
    );
  }
  return stdout;
};

export const encodeImage = async (
  ffmpeg: string,
  input: ArrayBufferView | Buffer,
  width: number,
  height: number,
  outputPath: string,
  options: { bitDepth?: 8 | 10 | 16 } = {},
): Promise<void> => {
  const bitDepth = options.bitDepth ?? 8;
  const pixFmt = bitDepth > 8 ? 'rgba64le' : 'rgba';
  const expectedBytes = width * height * 4 * (bitDepth > 8 ? 2 : 1);
  const buffer = Buffer.isBuffer(input)
    ? input
    : Buffer.from(input.buffer, input.byteOffset ?? 0, input.byteLength);
  if (buffer.byteLength !== expectedBytes) {
    throw new Error(
      `encodeImage expected ${expectedBytes} bytes for ${width}x${height} (bitDepth=${bitDepth}), received ${buffer.byteLength}`,
    );
  }
  const args = [
    '-v',
    'error',
    '-y',
    '-f',
    'rawvideo',
    '-pix_fmt',
    pixFmt,
    '-s',
    `${width}x${height}`,
    '-i',
    '-',
    outputPath,
  ];
  await runCommand(ffmpeg, args, { input: buffer });
};
