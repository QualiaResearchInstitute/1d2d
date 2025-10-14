import type { SceneGraphState, TimelineState } from './types.js';
import type { AppAction } from './AppState.js';

type SceneRuntimeSnapshot = {
  readonly scene: SceneGraphState;
  readonly timeline: TimelineState;
};

type SceneRuntimeListener = (snapshot: SceneRuntimeSnapshot) => void;

const listeners = new Set<SceneRuntimeListener>();

let lastSnapshot: SceneRuntimeSnapshot | undefined;
let dispatchRef: ((action: AppAction) => void) | undefined;

export const sceneRuntimeBridge = {
  publish(snapshot: SceneRuntimeSnapshot) {
    lastSnapshot = snapshot;
    listeners.forEach((listener) => listener(snapshot));
  },
  subscribe(listener: SceneRuntimeListener): () => void {
    listeners.add(listener);
    if (lastSnapshot) {
      listener(lastSnapshot);
    }
    return () => {
      listeners.delete(listener);
    };
  },
  getSnapshot(): SceneRuntimeSnapshot | undefined {
    return lastSnapshot;
  },
  registerDispatch(dispatch: (action: AppAction) => void) {
    dispatchRef = dispatch;
  },
  dispatch(action: AppAction) {
    dispatchRef?.(action);
  },
};
