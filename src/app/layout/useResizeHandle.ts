import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent, KeyboardEvent as ReactKeyboardEvent } from 'react';

type ResizeOrientation = 'vertical' | 'horizontal';
type ResizeDirection = 'positive' | 'negative';

export interface UseResizeHandleOptions {
  readonly orientation: ResizeOrientation;
  readonly direction: ResizeDirection;
  readonly size: number;
  readonly minSize: number;
  readonly maxSize?: number;
  readonly onSizeChange: (size: number) => void;
}

export interface ResizeHandle {
  readonly isDragging: boolean;
  readonly onPointerDown: (event: React.PointerEvent<HTMLDivElement>) => void;
  readonly onKeyDown: (event: ReactKeyboardEvent<HTMLDivElement>) => void;
}

const clamp = (value: number, min: number, max?: number) => {
  const lowerBounded = Math.max(min, value);
  return typeof max === 'number' ? Math.min(lowerBounded, max) : lowerBounded;
};

export function useResizeHandle(options: UseResizeHandleOptions): ResizeHandle {
  const { orientation, direction, size, minSize, maxSize, onSizeChange } = options;
  const [isDragging, setIsDragging] = useState(false);
  const startCoordinateRef = useRef(0);
  const startSizeRef = useRef(size);
  const latestSizeRef = useRef(size);
  const pointerIdRef = useRef<number | null>(null);

  useEffect(() => {
    latestSizeRef.current = size;
  }, [size]);

  const handlePointerMove = useCallback(
    (event: PointerEvent) => {
      // Prevent text selection or inadvertent browser gestures while resizing.
      event.preventDefault();
      if (pointerIdRef.current !== null && event.pointerId !== pointerIdRef.current) {
        return;
      }

      const currentCoordinate = orientation === 'vertical' ? event.clientX : event.clientY;
      const deltaRaw = currentCoordinate - startCoordinateRef.current;
      const delta = direction === 'positive' ? deltaRaw : -deltaRaw;
      const nextSize = clamp(startSizeRef.current + delta, minSize, maxSize);
      if (nextSize !== latestSizeRef.current) {
        latestSizeRef.current = nextSize;
        onSizeChange(nextSize);
      }
    },
    [direction, maxSize, minSize, onSizeChange, orientation],
  );

  const handlePointerUp = useCallback(
    (event: PointerEvent) => {
      if (pointerIdRef.current !== null && event.pointerId !== pointerIdRef.current) {
        return;
      }
      setIsDragging(false);
      pointerIdRef.current = null;
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    },
    [handlePointerMove],
  );

  useEffect(() => {
    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };
  }, [handlePointerMove, handlePointerUp]);

  const onPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      event.preventDefault();
      setIsDragging(true);
      pointerIdRef.current = event.pointerId;
      startCoordinateRef.current = orientation === 'vertical' ? event.clientX : event.clientY;
      startSizeRef.current = latestSizeRef.current;
      event.currentTarget.setPointerCapture(event.pointerId);
      window.addEventListener('pointermove', handlePointerMove);
      window.addEventListener('pointerup', handlePointerUp);
    },
    [handlePointerMove, handlePointerUp, orientation],
  );

  const handleKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      const { key } = event;
      let delta = 0;
      if (orientation === 'vertical') {
        if (key === 'ArrowLeft') {
          delta = -16;
        } else if (key === 'ArrowRight') {
          delta = 16;
        }
      } else if (key === 'ArrowUp') {
        delta = -16;
      } else if (key === 'ArrowDown') {
        delta = 16;
      }

      if (key === 'Home') {
        event.preventDefault();
        const target = clamp(minSize, minSize, maxSize);
        if (target !== latestSizeRef.current) {
          latestSizeRef.current = target;
          onSizeChange(target);
        }
        return;
      }

      if (key === 'End') {
        event.preventDefault();
        const target = clamp(
          typeof maxSize === 'number' ? maxSize : latestSizeRef.current,
          minSize,
          maxSize,
        );
        if (target !== latestSizeRef.current) {
          latestSizeRef.current = target;
          onSizeChange(target);
        }
        return;
      }

      if (delta === 0) {
        return;
      }

      if (direction === 'negative') {
        delta = -delta;
      }

      event.preventDefault();
      const nextSize = clamp(latestSizeRef.current + delta, minSize, maxSize);
      if (nextSize !== latestSizeRef.current) {
        latestSizeRef.current = nextSize;
        onSizeChange(nextSize);
      }
    },
    [direction, maxSize, minSize, onSizeChange, orientation],
  );

  return useMemo(
    () => ({
      isDragging,
      onPointerDown,
      onKeyDown: handleKeyDown,
    }),
    [handleKeyDown, isDragging, onPointerDown],
  );
}
